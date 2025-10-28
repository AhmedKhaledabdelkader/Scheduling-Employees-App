
"""
Dynamic Staff Scheduling Engine - Production Ready
===============================================

A comprehensive staff scheduling optimization tool with MILP/CP-SAT solver,
interactive Streamlit interface, and executive-grade reporting.

INSTALLATION & SETUP:
pip install streamlit pandas numpy openpyxl ortools pulp plotly seaborn matplotlib

USAGE:
streamlit run main.py

The app will start on http://localhost:8501

DATA CONTRACT:
Single Excel file 'input_data.xlsx' with sheets:
- employees: employee_id, name, hourly_rate, contract_min_hours, contract_max_hours, weekly_max_hours, skill_tags, employment_type, overtime_multiplier, max_consecutive_days
- shifts: shift_id, start_time, end_time, duration_hours, is_night
- demand: date, location_id, shift_id, required_staff
- availability: employee_id, date, shift_id, available
- eligibility: employee_id, location_id, eligible
- existing_schedule: employee_id, date, location_id, shift_id, published

Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
import pulp
from dataclasses import dataclass
from pathlib import Path
import io
# Create outputs directory first
Path('outputs').mkdir(exist_ok=True)
Path('outputs/charts').mkdir(exist_ok=True)

# Configure logging after directory creation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/app.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SchedulingConfig:
    """Configuration parameters for the scheduling optimization."""
    time_limit_seconds: int = 3600
    understaff_penalty: float = 1000.0
    overstaff_penalty: float = 100.0
    change_penalty: float = 50.0
    fairness_weight: float = 10.0
    solver_type: str = "CP-SAT"  # "CP-SAT" or "PULP"

class DataValidator:
    """Validates input data for scheduling optimization."""

    @staticmethod
    def validate_workbook(file_path: str) -> Dict[str, Any]:
        """Validate the Excel workbook structure and data."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                results['valid'] = False
                results['errors'].append(f"File not found: {file_path}")
                return results

            # Load all sheets
            xl_file = pd.ExcelFile(file_path)
            required_sheets = ['employees', 'shifts', 'demand', 'availability', 'existing_schedule', 'eligibility']

            # Check required sheets exist
            for sheet in required_sheets:
                if sheet not in xl_file.sheet_names:
                    results['valid'] = False
                    results['errors'].append(f"Missing required sheet: {sheet}")

            if not results['valid']:
                return results

            # Validate each sheet
            sheets_data = {}
            for sheet in required_sheets:
                sheets_data[sheet] = pd.read_excel(file_path, sheet_name=sheet)

            # Validate employees sheet
            emp_required = ['employee_id', 'name', 'hourly_rate', 'contract_min_hours',
                          'contract_max_hours', 'weekly_max_hours', 'skill_tags',
                          'employment_type', 'overtime_multiplier', 'max_consecutive_days']
            DataValidator._validate_columns(sheets_data['employees'], emp_required, 'employees', results)

            # Validate shifts sheet
            shift_required = ['shift_id', 'start_time', 'end_time', 'duration_hours', 'is_night']
            DataValidator._validate_columns(sheets_data['shifts'], shift_required, 'shifts', results)

            # Validate demand sheet
            demand_required = ['date', 'location_id', 'shift_id', 'required_staff']
            DataValidator._validate_columns(sheets_data['demand'], demand_required, 'demand', results)

            # Validate availability sheet
            avail_required = ['employee_id', 'date', 'shift_id', 'available']
            DataValidator._validate_columns(sheets_data['availability'], avail_required, 'availability', results)

            # Validate existing_schedule sheet
            sched_required = ['employee_id', 'date', 'location_id', 'shift_id', 'published']
            DataValidator._validate_columns(sheets_data['existing_schedule'], sched_required, 'existing_schedule', results)

            # Validate eligibility sheet
            elig_required = ['employee_id', 'location_id', 'eligible']
            DataValidator._validate_columns(sheets_data['eligibility'], elig_required, 'eligibility', results)

            # Data consistency checks
            DataValidator._validate_data_consistency(sheets_data, results)

            # Generate summary
            results['summary'] = {
                'employees': len(sheets_data['employees']),
                'shifts': len(sheets_data['shifts']),
                'locations': sheets_data['demand']['location_id'].nunique(),
                'date_range': f"{sheets_data['demand']['date'].min()} to {sheets_data['demand']['date'].max()}",
                'total_demand': sheets_data['demand']['required_staff'].sum()
            }

        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Error reading file: {str(e)}")

        return results

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_cols: List[str], sheet_name: str, results: Dict):
        """Validate that required columns exist in a dataframe."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Sheet '{sheet_name}' missing columns: {missing_cols}")

    @staticmethod
    def _validate_data_consistency(sheets: Dict[str, pd.DataFrame], results: Dict):
        """Validate data consistency across sheets."""
        # Check employee IDs are consistent
        emp_ids = set(sheets['employees']['employee_id'])
        avail_emp_ids = set(sheets['availability']['employee_id'])
        sched_emp_ids = set(sheets['existing_schedule']['employee_id'])

        missing_in_avail = avail_emp_ids - emp_ids
        missing_in_sched = sched_emp_ids - emp_ids

        if missing_in_avail:
            results['warnings'].append(f"Employee IDs in availability not in employees: {missing_in_avail}")
        if missing_in_sched:
            results['warnings'].append(f"Employee IDs in schedule not in employees: {missing_in_sched}")

        # Check shift IDs are consistent
        shift_ids = set(sheets['shifts']['shift_id'])
        demand_shift_ids = set(sheets['demand']['shift_id'])
        avail_shift_ids = set(sheets['availability']['shift_id'])

        if not demand_shift_ids.issubset(shift_ids):
            results['errors'].append("Some shift IDs in demand not found in shifts")
            results['valid'] = False

        if not avail_shift_ids.issubset(shift_ids):
            results['errors'].append("Some shift IDs in availability not found in shifts")
            results['valid'] = False

        # Check eligibility data
        elig_emp_ids = set(sheets['eligibility']['employee_id'])
        elig_loc_ids = set(sheets['eligibility']['location_id'])
        demand_loc_ids = set(sheets['demand']['location_id'])

        if not elig_emp_ids.issubset(emp_ids):
            results['warnings'].append(f"Employee IDs in eligibility not in employees: {elig_emp_ids - emp_ids}")

        if not demand_loc_ids.issubset(elig_loc_ids):
            results['warnings'].append("Some locations in demand are not covered in eligibility sheet")

class SchedulingSolver:
    """Core optimization engine for staff scheduling."""

    def __init__(self, config: SchedulingConfig):
        self.config = config
        self.employees = None
        self.shifts = None
        self.demand = None
        self.availability = None
        self.existingschedule = None
        self.eligibility = None


    def load_data(self, file_path: str):
        """Load scheduling data from Excel file."""
        try:
            # Read the Excel file with window.fs API
            file_data = window.fs.readFile(file_path)

            # Convert to BytesIO for pandas
            excel_buffer = BytesIO(file_data)

            # Read all sheets
            excel_file = pd.ExcelFile(excel_buffer)
            print(f"Available sheets: {excel_file.sheet_names}")

            self.employees = pd.read_excel(excel_buffer, sheet_name='employees')
            self.shifts = pd.read_excel(excel_buffer, sheet_name='shifts')
            self.demand = pd.read_excel(excel_buffer, sheet_name='demand')
            self.availability = pd.read_excel(excel_buffer, sheet_name='availability')
            self.existing_schedule = pd.read_excel(excel_buffer, sheet_name='existing_schedule')
            self.eligibility = pd.read_excel(excel_buffer, sheet_name='eligibility')


            # Debug: Print data info
            print("Employees shape:", self.employees.shape)
            print("Employees columns:", self.employees.columns.tolist())
            print("Sample employees:\n", self.employees.head())

            print("Demand shape:", self.demand.shape)
            print("Demand columns:", self.demand.columns.tolist())
            print("Sample demand:\n", self.demand.head())

            print("Availability shape:", self.availability.shape)
            print("Sample availability:\n", self.availability.head())

        except Exception as e:
            print(f"Error reading with window.fs, falling back to regular file reading: {e}")
            # Fallback to regular file reading
            self.employees = pd.read_excel(file_path, sheet_name='employees')
            self.shifts = pd.read_excel(file_path, sheet_name='shifts')
            self.demand = pd.read_excel(file_path, sheet_name='demand')
            self.availability = pd.read_excel(file_path, sheet_name='availability')
            self.existing_schedule = pd.read_excel(file_path, sheet_name='existing_schedule')
            self.eligibility = pd.read_excel(file_path, sheet_name='eligibility')


        # Convert date columns to datetime - handle different formats
        try:
            self.demand['date'] = pd.to_datetime(self.demand['date'], errors='coerce')
            self.availability['date'] = pd.to_datetime(self.availability['date'], errors='coerce')
            self.existing_schedule['date'] = pd.to_datetime(self.existing_schedule['date'], errors='coerce')
        except Exception as e:
            print(f"Error converting dates: {e}")

        # Parse skill tags
        if 'skill_tags' in self.employees.columns:
            self.employees['skill_tags'] = self.employees['skill_tags'].fillna('').astype(str)

        # Clean up data - remove NaN rows
        self.demand = self.demand.dropna(subset=['date', 'location_id', 'shift_id'])
        self.availability = self.availability.dropna(subset=['employee_id', 'date', 'shift_id'])
        self.eligibility = self.eligibility.dropna(subset=['employee_id', 'location_id'])

        # Debug final data
        print("Final demand rows:", len(self.demand))
        print("Final availability rows:", len(self.availability))
        print("Date range in demand:", self.demand['date'].min(), "to", self.demand['date'].max())

    def optimize(self) -> Dict[str, Any]:
        """Run the optimization and return results."""
        logger.info("Starting optimization...")
        start_time = datetime.now()

        try:
            if self.config.solver_type == "CP-SAT":
                result = self._solve_cpsat()
            else:
                result = self._solve_pulp()

            solve_time = (datetime.now() - start_time).total_seconds()
            result['solve_time'] = solve_time

            logger.info(f"Optimization completed in {solve_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'solve_time': (datetime.now() - start_time).total_seconds()
            }

    def _solve_cpsat(self) -> Dict[str, Any]:
        """Solve using CP-SAT solver."""
        model = cp_model.CpModel()

        # Get unique values
        employees = list(self.employees['employee_id'])
        shifts = list(self.shifts['shift_id'])
        dates = sorted(list(self.demand['date'].unique()))
        locations = list(self.demand['location_id'].unique())

        print(f"Problem size: {len(employees)} employees, {len(shifts)} shifts, {len(dates)} dates, {len(locations)} locations")

        # Check if problem is too large
        total_possible_variables = len(employees) * len(shifts) * len(dates) * len(locations)
        if total_possible_variables > 50000:
            print(f"ERROR: Problem too large ({total_possible_variables} possible variables)")
            return {
                'status': 'ERROR',
                'error': f'Problem too large: {total_possible_variables} variables exceed limit of 50,000',
                'solve_time': 0
            }


        # Check feasibility basics
        total_demand_hours = 0
        for _, row in self.demand.iterrows():
            shift_duration = self.shifts[self.shifts['shift_id'] == row['shift_id']]['duration_hours'].iloc[0]
            total_demand_hours += row['required_staff'] * shift_duration

        total_available_capacity = 0
        for _, emp in self.employees.iterrows():
            total_available_capacity += emp['weekly_max_hours'] * (len(dates) // 7 + 1)

        print(f"Total demand hours: {total_demand_hours}")
        print(f"Total available capacity: {total_available_capacity}")

        if total_demand_hours > total_available_capacity:
            print("WARNING: Demand exceeds total capacity - expect understaffing")

        # Decision variables
        # Decision variables - only create where employee is available AND eligible
        x = {}
        for i in employees:
            for s in shifts:
                for t in dates:
                    for l in locations:
                        # Only create variables where there's actual demand
                        demand_exists = not self.demand[
                            (self.demand['shift_id'] == s) &
                            (self.demand['date'] == t) &
                            (self.demand['location_id'] == l)
                        ].empty

                        # AND employee is available
                        is_available = not self.availability[
                            (self.availability['employee_id'] == i) &
                            (self.availability['date'] == t) &
                            (self.availability['shift_id'] == s) &
                            (self.availability['available'] == 1)
                        ].empty

                        # AND employee is eligible for this location
                        is_eligible = not self.eligibility[
                            (self.eligibility['employee_id'] == i) &
                            (self.eligibility['location_id'] == l) &
                            (self.eligibility['eligible'] == 1)
                        ].empty

                        if demand_exists and is_available and is_eligible:
                            x[i,s,t,l] = model.NewBoolVar(f'x_{i}_{s}_{t.strftime("%Y%m%d")}_{l}')


        print(f"Created {len(x)} decision variables")

        # Understaffing variables
        u = {}
        for _, demand_row in self.demand.iterrows():
            s, t, l = demand_row['shift_id'], demand_row['date'], demand_row['location_id']
            required = int(demand_row['required_staff'])
            if required > 0:
                u[s,t,l] = model.NewIntVar(0, required, f'u_{s}_{t.strftime("%Y%m%d")}_{l}')

        # Overstaffing variables - limit maximum overstaffing
        o = {}
        for _, demand_row in self.demand.iterrows():
            s, t, l = demand_row['shift_id'], demand_row['date'], demand_row['location_id']
            required = int(demand_row['required_staff'])
            if required > 0:
                # Limit overstaffing to max 3 extra staff per shift
                o[s,t,l] = model.NewIntVar(0, min(3, required), f'o_{s}_{t.strftime("%Y%m%d")}_{l}')


        print(f"Created {len(u)} understaffing variables")
        print(f"Created {len(o)} overstaffing variables")

        # Overtime variables and constraints
        ot = {}
        overtime_constraint_count = 0
        for i in employees:
            emp_data = self.employees[self.employees['employee_id'] == i].iloc[0]
            max_regular = int(emp_data['contract_max_hours'])
            max_total = int(emp_data['weekly_max_hours'])

            ot[i] = model.NewIntVar(0, max_total - max_regular, f'ot_{i}')

            # Calculate total hours for this employee
            total_hours_terms = []
            for s in shifts:
                shift_hours = int(self.shifts[self.shifts['shift_id'] == s]['duration_hours'].iloc[0])
                for t in dates:
                    for l in locations:
                        if (i,s,t,l) in x:
                            total_hours_terms.append(x[i,s,t,l] * shift_hours)

            if total_hours_terms:
                # Overtime constraint: ot_i >= total_hours - max_regular
                model.Add(ot[i] >= sum(total_hours_terms) - max_regular)
                model.Add(ot[i] >= 0)
                overtime_constraint_count += 1

        print(f"Added {overtime_constraint_count} overtime constraints")

        # Change variables - track differences from existing schedule
        chg = {}
        existing_assignments = set()

        # Check if existing_schedule has actual data
        has_existing_schedule = (
            not self.existing_schedule.empty and
            len(self.existing_schedule) > 0 and
            not self.existing_schedule.dropna(how='all').empty
        )

        if has_existing_schedule:
            print("Existing schedule found - enabling change tracking...")
            # Build set of existing assignments
            for _, row in self.existing_schedule.iterrows():
                try:
                    existing_assignments.add((row['employee_id'], row['date'], row['location_id'], row['shift_id']))
                except Exception:
                    continue
        else:
            print("No existing schedule - skipping change tracking entirely")

        # Create change variables and constraints
        change_constraint_count = 0
        if has_existing_schedule:
            for i in employees:
                for s in shifts:
                    for t in dates:
                        for l in locations:
                            if (i,s,t,l) in x:
                                # Create continuous variable for absolute difference (not binary)
                                chg[i,s,t,l] = model.NewIntVar(0, 1, f"chg_{i}_{s}_{t.strftime('%Y%m%d')}_{l}")
                                was_assigned = 1 if (i, t, s, l) in existing_assignments else 0

                                # TWO constraints to capture absolute difference: |x - x^(0)|
                                model.Add(chg[i,s,t,l] >= x[i,s,t,l] - was_assigned)
                                model.Add(chg[i,s,t,l] >= was_assigned - x[i,s,t,l])
                                change_constraint_count += 2
            print(f"Created {len(chg)} change variables with {change_constraint_count} constraints")
        else:
            print("Change tracking skipped (no existing schedule)")

        # Constraints
        print("Adding constraints...")

        # Coverage constraints
        constraint_count = 0
        for _, demand_row in self.demand.iterrows():
            s, t, l, required = demand_row['shift_id'], demand_row['date'], demand_row['location_id'], int(demand_row['required_staff'])

            if required > 0:
                # Get all employees who could work this shift
                available_vars = []
                for i in employees:
                    if (i,s,t,l) in x:
                        # Check availability
                        is_available = self.availability[
                            (self.availability['employee_id'] == i) &
                            (self.availability['date'] == t) &
                            (self.availability['shift_id'] == s) &
                            (self.availability['available'] == 1)
                        ]

                        if not is_available.empty and len(is_available) > 0:
                            available_vars.append(x[i,s,t,l])


                if available_vars:
                    # Coverage constraint: sum(assignments) + u - o = required
                    if (s,t,l) in u and (s,t,l) in o:
                        model.Add(sum(available_vars) - o[s,t,l] + u[s,t,l] == required)
                    constraint_count += 1
                else:
                    print(f"WARNING: No available employees for {s} on {t} at {l}")
                    # This demand cannot be met - add large understaffing
                    if (s,t,l) in u:
                        model.Add(u[s,t,l] == required)

        print(f"Added {constraint_count} coverage constraints")

        # Force at least one real staff if demand > 0 constraint
        min_staffing_count = 0
        for _, demand_row in self.demand.iterrows():
            s, t, l, required = demand_row['shift_id'], demand_row['date'], demand_row['location_id'], int(demand_row['required_staff'])
            if required > 0:
                # Get all employees who could work this shift
                available_vars = []
                for i in employees:
                    if (i,s,t,l) in x:
                        # Check availability
                        is_available = self.availability[
                            (self.availability['employee_id'] == i) &
                            (self.availability['date'] == t) &
                            (self.availability['shift_id'] == s) &
                            (self.availability['available'] == 1)
                        ]
                        if not is_available.empty and len(is_available) > 0:
                            available_vars.append(x[i,s,t,l])


                # Force at least 1 employee if demand > 0
                if available_vars:
                    model.Add(sum(available_vars) >= 1)
                    min_staffing_count += 1

        print(f"Added {min_staffing_count} minimum staffing constraints (at least 1 employee per demanded shift)")

        # Employee constraints - one shift per day maximum
        emp_constraint_count = 0
        for i in employees:
            for t in dates:
                day_vars = []
                for s in shifts:
                    for l in locations:
                        if (i,s,t,l) in x:
                            day_vars.append(x[i,s,t,l])

                if day_vars:
                    model.Add(sum(day_vars) <= 1)
                    emp_constraint_count += 1

        print(f"Added {emp_constraint_count} employee daily constraints")


        # Apply employee constraints
        self._add_employee_constraints_cpsat(model, x, employees, shifts, dates, locations)
        self._add_availability_constraints_cpsat(model, x, employees, shifts, dates, locations)



        # Add maximum consecutive days constraints
        consecutive_constraint_count = 0
        for i in employees:
            emp_data = self.employees[self.employees['employee_id'] == i].iloc[0]
            max_consecutive = int(emp_data['max_consecutive_days'])

            # For each possible sequence of max_consecutive+1 days
            for start_idx in range(len(dates) - max_consecutive):
                consecutive_days = dates[start_idx:start_idx + max_consecutive + 1]
                # Sum of work assignments over max_consecutive+1 days must be <= max_consecutive
                consecutive_sum = sum(
                    x[i,s,t,l]
                    for s in shifts for t in consecutive_days for l in locations
                    if (i,s,t,l) in x
                )
                model.Add(consecutive_sum <= max_consecutive)
                consecutive_constraint_count += 1
        print(f"Added {consecutive_constraint_count} consecutive days constraints")

        # Add Forbidden consecutive shifts constraint
        # Check if shifts DataFrame has 'forbidden_next' column
        forbidden_constraint_count = 0
        if 'forbidden_next' in self.shifts.columns:
            print("Adding forbidden consecutive shift constraints...")
            for i in employees:
                for t_idx in range(len(dates) - 1):  # All days except the last
                    t = dates[t_idx]
                    t_next = dates[t_idx + 1]

                    # For each shift type
                    for _, shift_row in self.shifts.iterrows():
                        s = shift_row['shift_id']
                        forbidden_next_list = shift_row.get('forbidden_next', '')

                        # Skip if no forbidden next shifts defined
                        if pd.isna(forbidden_next_list) or forbidden_next_list == '':
                            continue

                        # Parse forbidden next shifts (comma-separated)
                        forbidden_next = [fn.strip() for fn in str(forbidden_next_list).split(',')]

                        # For each forbidden next shift
                        for s_prime in forbidden_next:
                            if s_prime not in shifts:
                                continue

                            # Sum over all locations for day t with shift s
                            day_t_vars = [x[i,s,t,l] for l in locations if (i,s,t,l) in x]

                            # Sum over all locations for day t+1 with shift s'
                            day_t_next_vars = [x[i,s_prime,t_next,l] for l in locations if (i,s_prime,t_next,l) in x]

                            # Constraint: sum(x[i,s,t,l]) + sum(x[i,s',t+1,l']) <= 1
                            if day_t_vars and day_t_next_vars:
                                model.Add(sum(day_t_vars) + sum(day_t_next_vars) <= 1)
                                forbidden_constraint_count += 1

            print(f"Added {forbidden_constraint_count} forbidden consecutive shift constraints")
        else:
            print("WARNING: No 'forbidden_next' column in shifts sheet - skipping forbidden consecutive shifts constraint")

        # Minimum hours constraints
        min_hours_constraint_count = 0
        for i in employees:
            emp_data = self.employees[self.employees['employee_id'] == i].iloc[0]
            min_hours = int(emp_data['contract_min_hours'])

            # Calculate total hours for this employee
            total_hours_terms = []
            for s in shifts:
                shift_hours = int(self.shifts[self.shifts['shift_id'] == s]['duration_hours'].iloc[0])
                for t in dates:
                    for l in locations:
                        if (i,s,t,l) in x:
                            total_hours_terms.append(x[i,s,t,l] * shift_hours)

            if total_hours_terms:
                model.Add(sum(total_hours_terms) >= min_hours)
                min_hours_constraint_count += 1

        print(f"Added {min_hours_constraint_count} minimum hours constraints")

        # Add Maximum Hours Constraint
        max_hours_constraint_count = 0
        for i in employees:
            emp_data = self.employees[self.employees['employee_id'] == i].iloc[0]
            max_hours = int(emp_data['contract_max_hours'])

            # Calculate total hours for this employee
            total_hours_terms = []
            for s in shifts:
                shift_hours = int(self.shifts[self.shifts['shift_id'] == s]['duration_hours'].iloc[0])
                for t in dates:
                    for l in locations:
                        if (i,s,t,l) in x:
                            total_hours_terms.append(x[i,s,t,l] * shift_hours)

            if total_hours_terms:
                model.Add(sum(total_hours_terms) <= max_hours)
                max_hours_constraint_count += 1

        print(f"Added {max_hours_constraint_count} maximum hours constraints")


        # Complete objective function matching the mathematical model
        objective_terms = []

        # 1. Regular labor costs
        labor_cost_terms = []
        for i in employees:
            emp_data = self.employees[self.employees['employee_id'] == i].iloc[0]
            hourly_rate = int(emp_data['hourly_rate'] * 100)  # Scale for integer arithmetic

            for s in shifts:
                shift_hours = int(self.shifts[self.shifts['shift_id'] == s]['duration_hours'].iloc[0])
                for t in dates:
                    for l in locations:
                        if (i,s,t,l) in x:
                            cost = hourly_rate * shift_hours
                            labor_cost_terms.append(x[i,s,t,l] * cost)

        # 2. Overtime costs
        overtime_cost_terms = []
        for i in employees:
            emp_data = self.employees[self.employees['employee_id'] == i].iloc[0]
            # r_i = (alpha_i - 1) * w_i : overtime premium per hour
            overtime_premium = int(emp_data['hourly_rate'] * (emp_data['overtime_multiplier'] - 1) * 100)
            if i in ot:
                overtime_cost_terms.append(ot[i] * overtime_premium)


        # 3. Understaffing penalties
        understaff_terms = []
        for key in u:
            understaff_terms.append(u[key] * int(self.config.understaff_penalty))

        # 4. Overstaffing penalties
        overstaff_terms = []
        for key in o:
            overstaff_terms.append(o[key] * int(self.config.overstaff_penalty))

        # 5. Change penalties (only if change tracking is enabled)
        change_terms = []
        if has_existing_schedule and chg:
            for key in chg:
                change_terms.append(chg[key] * int(self.config.change_penalty))

        # Combine all objective terms (change terms only if applicable)
        if has_existing_schedule:
            all_terms = labor_cost_terms + overtime_cost_terms + understaff_terms + overstaff_terms + change_terms
        else:
            all_terms = labor_cost_terms + overtime_cost_terms + understaff_terms + overstaff_terms

        if all_terms:
            model.Minimize(sum(all_terms))
            print(f"Objective includes: {len(labor_cost_terms)} labor + {len(overtime_cost_terms)} overtime + {len(understaff_terms)} understaff + {len(overstaff_terms)} overstaff + {len(change_terms)} change terms")
        else:
            print("ERROR: No objective terms created")
            return {'status': 'ERROR', 'error': 'No objective function'}

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.time_limit_seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.log_search_progress = True


        print("Starting solve...")
        status = solver.Solve(model)
        print(f"Solve completed with status: {status}")

        # Process results
        return self._process_cpsat_results(solver, status, x, u, o, ot, chg, employees, shifts, dates, locations)

    def _add_coverage_constraints_cpsat(self, model, x, u, employees, shifts, dates, locations):
        """Add coverage constraints to CP-SAT model."""
        for _, row in self.demand.iterrows():
            s, t, l, required = row['shift_id'], row['date'], row['location_id'], row['required_staff']

            assigned = sum(x[i,s,t,l] for i in employees
                          if (i,s,t,l) in x)

            if (s,t,l) in u:
                model.Add(assigned + u[s,t,l] >= required)
            else:
                model.Add(assigned >= required)

    def _add_employee_constraints_cpsat(self, model, x, employees, shifts, dates, locations):
        """Add employee-specific constraints."""
        # This is already handled in the main solve function - do nothing to avoid duplicates
        pass

    def _add_availability_constraints_cpsat(self, model, x, employees, shifts, dates, locations):
        """Add availability constraints."""
        for _, row in self.availability.iterrows():
            i, t, s, available = row['employee_id'], row['date'], row['shift_id'], row['available']

            if available == 0:  # Not available
                for l in locations:
                    if (i,s,t,l) in x:
                        model.Add(x[i,s,t,l] == 0)

    def _add_change_constraints_cpsat(self, model, x, chg, employees, shifts, dates, locations):
        """Add change tracking constraints."""
        existing_assignments = set()
        for _, row in self.existing_schedule.iterrows():
            i, t, l, s = row['employee_id'], row['date'], row['location_id'], row['shift_id']
            existing_assignments.add((i,s,t,l))

        for i in employees:
            for s in shifts:
                for t in dates:
                    for l in locations:
                        if (i,s,t,l) in x and (i,s,t,l) in chg:
                            if (i,s,t,l) in existing_assignments:
                                # Was assigned, track if unassigned
                                model.Add(chg[i,s,t,l] >= 1 - x[i,s,t,l])
                            else:
                                # Was not assigned, track if assigned
                                model.Add(chg[i,s,t,l] >= x[i,s,t,l])

    def _add_overtime_constraints_cpsat(self, model, x, ot, employees, shifts, dates, locations):
        """Add overtime calculation constraints."""
        for i in employees:
            emp_data = self.employees[self.employees['employee_id'] == i].iloc[0]
            regular_hours = emp_data['contract_max_hours']

            total_hours = sum(
                x[i,s,t,l] * self.shifts[self.shifts['shift_id'] == s]['duration_hours'].iloc[0]
                for s in shifts for t in dates for l in locations
                if (i,s,t,l) in x and not self.shifts[self.shifts['shift_id'] == s].empty
            )

            model.Add(ot[i] >= total_hours - regular_hours)
            model.Add(ot[i] >= 0)

    def _set_objective_cpsat(self, model, x, u, ot, chg, employees, shifts, dates, locations):
        """Set the objective function for CP-SAT."""
        objective_terms = []

        # Regular labor cost
        for i in employees:
            emp_data = self.employees[self.employees['employee_id'] == i].iloc[0]
            hourly_rate = emp_data['hourly_rate']

            for s in shifts:
                shift_hours = self.shifts[self.shifts['shift_id'] == s]['duration_hours'].iloc[0]
                for t in dates:
                    for l in locations:
                        if (i,s,t,l) in x:
                            objective_terms.append(x[i,s,t,l] * int(hourly_rate * shift_hours))

        # Overtime cost
        for i in employees:
            emp_data = self.employees[self.employees['employee_id'] == i].iloc[0]
            # r_i = (alpha_i - 1) * w_i : cost per overtime hour (premium only)
            overtime_premium = emp_data['hourly_rate'] * (emp_data['overtime_multiplier'] - 1)
            if i in ot:
                objective_terms.append(ot[i] * int(overtime_premium))

        # Understaffing penalty
        for key in u:
            objective_terms.append(u[key] * int(self.config.understaff_penalty))

        # Change penalty
        for key in chg:
            objective_terms.append(chg[key] * int(self.config.change_penalty))

        model.Minimize(sum(objective_terms))

    def _process_cpsat_results(self, solver, status, x, u, o, ot, chg, employees, shifts, dates, locations):
        """Process CP-SAT solver results."""
        status_map = {
            cp_model.OPTIMAL: 'OPTIMAL',
            cp_model.FEASIBLE: 'FEASIBLE',
            cp_model.INFEASIBLE: 'INFEASIBLE',
            cp_model.UNKNOWN: 'TIMEOUT'
        }

        result = {
            'status': status_map.get(status, 'UNKNOWN'),
            'objective_value': solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else 0,
            'assignments': [],
            'understaffing': [],
            'overstaffing': [],
            'overtime': {},
            'changes': []
        }

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Extract assignments
            for i in employees:
                for s in shifts:
                    for t in dates:
                        for l in locations:
                            if (i,s,t,l) in x and solver.Value(x[i,s,t,l]) == 1:
                                result['assignments'].append({
                                    'employee_id': i,
                                    'date': t,
                                    'location_id': l,
                                    'shift_id': s,
                                    'hours': self.shifts[self.shifts['shift_id'] == s]['duration_hours'].iloc[0]
                                })

            # Extract understaffing
            for key in u:
                understaffing_val = solver.Value(u[key])
                if understaffing_val > 0:
                    s, t, l = key
                    result['understaffing'].append({
                        'shift_id': s,
                        'date': t,
                        'location_id': l,
                        'shortage': understaffing_val
                    })

            # Extract overtime
            for i in employees:
                if i in ot:
                    ot_hours = solver.Value(ot[i])
                    if ot_hours > 0:
                        result['overtime'][i] = ot_hours

            # Extract overstaffing
            for key in o:
                overstaffing_val = solver.Value(o[key])
                if overstaffing_val > 0:
                    s, t, l = key
                    result['overstaffing'].append({
                        'shift_id': s,
                        'date': t,
                        'location_id': l,
                        'excess': overstaffing_val
                    })

        return result

    def _solve_pulp(self) -> Dict[str, Any]:
        """Fallback solver using PuLP."""
        prob = pulp.LpProblem("StaffScheduling", pulp.LpMinimize)

        employees = list(self.employees['employee_id'])
        shifts = list(self.shifts['shift_id'])
        dates = sorted(list(self.demand['date'].unique()))
        locations = list(self.demand['location_id'].unique())

        # Decision variables
        x = pulp.LpVariable.dicts("assign",
                                 [(i,s,t,l) for i in employees for s in shifts
                                  for t in dates for l in locations],
                                 cat='Binary')

        # Understaffing variables
        u = pulp.LpVariable.dicts("understaff",
                                 [(s,t,l) for s in shifts for t in dates for l in locations],
                                 lowBound=0)

        # Overstaffing variables
        o = pulp.LpVariable.dicts("overstaff",
                                 [(s,t,l) for s in shifts for t in dates for l in locations],
                                 lowBound=0)

        # Objective function
        objective_terms = []

        # Labor costs
        for i in employees:
            emp_data = self.employees[self.employees['employee_id'] == i].iloc[0]
            hourly_rate = emp_data['hourly_rate']
            for s in shifts:
                shift_hours = self.shifts[self.shifts['shift_id'] == s]['duration_hours'].iloc[0]
                for t in dates:
                    for l in locations:
                        objective_terms.append(x[i,s,t,l] * hourly_rate * shift_hours)

        # Penalties
        for s in shifts:
            for t in dates:
                for l in locations:
                    objective_terms.append(u[s,t,l] * self.config.understaff_penalty)
                    objective_terms.append(o[s,t,l] * self.config.overstaff_penalty)

        prob += pulp.lpSum(objective_terms)

        # Coverage constraints
        for _, row in self.demand.iterrows():
            s, t, l, required = row['shift_id'], row['date'], row['location_id'], row['required_staff']
            prob += pulp.lpSum([x[i,s,t,l] for i in employees]) - o[s,t,l] + u[s,t,l] == required

        # One shift per day
        for i in employees:
            for t in dates:
                prob += pulp.lpSum([x[i,s,t,l] for s in shifts for l in locations]) <= 1

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(timeLimit=self.config.time_limit_seconds))

        result = {
            'status': pulp.LpStatus[prob.status],
            'objective_value': prob.objective.value() if prob.objective.value() is not None else 0,
            'assignments': [],
            'understaffing': [],
            'overstaffing': [],
            'overtime': {}
        }

        if prob.status == pulp.LpStatusOptimal:
            for i in employees:
                for s in shifts:
                    for t in dates:
                        for l in locations:
                            if x[i,s,t,l].value() == 1:
                                result['assignments'].append({
                                    'employee_id': i,
                                    'date': t,
                                    'location_id': l,
                                    'shift_id': s,
                                    'hours': self.shifts[self.shifts['shift_id'] == s]['duration_hours'].iloc[0]
                                })

            # Extract understaffing
            for s in shifts:
                for t in dates:
                    for l in locations:
                        if u[s,t,l].value() and u[s,t,l].value() > 0:
                            result['understaffing'].append({
                                'shift_id': s,
                                'date': t,
                                'location_id': l,
                                'shortage': u[s,t,l].value()
                            })

            # Extract overstaffing
            for s in shifts:
                for t in dates:
                    for l in locations:
                        if o[s,t,l].value() and o[s,t,l].value() > 0:
                            result['overstaffing'].append({
                                'shift_id': s,
                                'date': t,
                                'location_id': l,
                                'excess': o[s,t,l].value()
                            })

        return result

class SampleDataGenerator:
    """Generate sample data for demonstration."""

    @staticmethod
    def generate_sample_data() -> Dict[str, pd.DataFrame]:
        """Generate comprehensive sample dataset."""

        # Sample employees
        employees = pd.DataFrame({
            'employee_id': [f'EMP{i:03d}' for i in range(1, 21)],
            'name': [f'Employee {i}' for i in range(1, 21)],
            'hourly_rate': np.random.uniform(15, 25, 20),
            'contract_min_hours': [32] * 20,
            'contract_max_hours': [40] * 20,
            'weekly_max_hours': [48] * 20,
            'skill_tags': np.random.choice(['cashier', 'manager', 'stocker', 'cashier,stocker'], 20),
            'employment_type': np.random.choice(['full_time', 'part_time'], 20),
            'overtime_multiplier': [1.5] * 20,
            'max_consecutive_days': [5] * 20
        })

        # Sample shifts
        shifts = pd.DataFrame({
            'shift_id': ['MORNING', 'AFTERNOON', 'EVENING', 'NIGHT'],
            'start_time': ['06:00', '14:00', '18:00', '22:00'],
            'end_time': ['14:00', '18:00', '22:00', '06:00'],
            'duration_hours': [8, 4, 4, 8],
            'is_night': [0, 0, 0, 1]
        })

        # Sample demand - one week
        dates = pd.date_range('2024-01-01', periods=7, freq='D')
        locations = ['LOC001', 'LOC002', 'LOC003']

        demand_records = []
        for date in dates:
            for location in locations:
                for _, shift in shifts.iterrows():
                    # Higher demand on weekends
                    base_demand = 3 if date.weekday() < 5 else 5
                    if shift['shift_id'] == 'NIGHT':
                        base_demand = max(1, base_demand - 1)

                    demand_records.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'location_id': location,
                        'shift_id': shift['shift_id'],
                        'required_staff': base_demand + np.random.randint(-1, 2)
                    })

        demand = pd.DataFrame(demand_records)

        # Sample availability (most employees available most times)
        availability_records = []
        for _, emp in employees.iterrows():
            for date in dates:
                for _, shift in shifts.iterrows():
                    # 85% availability rate
                    available = 1 if np.random.random() > 0.15 else 0
                    # Night shifts less popular
                    if shift['shift_id'] == 'NIGHT' and np.random.random() > 0.6:
                        available = 0

                    availability_records.append({
                        'employee_id': emp['employee_id'],
                        'date': date.strftime('%Y-%m-%d'),
                        'shift_id': shift['shift_id'],
                        'available': available
                    })

        availability = pd.DataFrame(availability_records)

        # Sample existing schedule (sparse)
        existing_records = []
        for i in range(30):  # 30 existing assignments
            emp_id = np.random.choice(employees['employee_id'])
            date = np.random.choice(dates).strftime('%Y-%m-%d')
            location = np.random.choice(locations)
            shift = np.random.choice(shifts['shift_id'])

            existing_records.append({
                'employee_id': emp_id,
                'date': date,
                'location_id': location,
                'shift_id': shift,
                'published': 1
            })

        existing_schedule = pd.DataFrame(existing_records).drop_duplicates()

        return {
            'employees': employees,
            'shifts': shifts,
            'demand': demand,
            'availability': availability,
            'existing_schedule': existing_schedule
        }

    @staticmethod
    def save_sample_data(filename: str = 'input_data.xlsx'):
        """Save sample data to Excel file."""
        data = SampleDataGenerator.generate_sample_data()

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        return filename

class ReportGenerator:
    """Generate comprehensive reports and visualizations."""

    def __init__(self, solver: SchedulingSolver, results: Dict[str, Any]):
        self.solver = solver
        self.results = results
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def generate_all_outputs(self):
        """Generate all required output files."""
        try:
            if self.results.get('status') in ['OPTIMAL', 'FEASIBLE']:
                self.generate_schedule_excel()
                self.generate_payroll_excel()
                self.generate_changelog_excel()
                self.generate_charts()
                self.generate_aggregated_results_file()
            else:
                # For infeasible solutions, generate diagnostic report
                self.generate_diagnostic_report()

            # Always generate insights
            self.generate_insights_json()
            logger.info("All outputs generated successfully")
        except Exception as e:
            logger.error(f"Error generating outputs: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def generate_diagnostic_report(self):
        """Generate diagnostic report for infeasible solutions."""
        diagnostic_data = {
            'status': self.results.get('status', 'UNKNOWN'),
            'error': self.results.get('error', 'No error message'),
            'problem_analysis': self._analyze_infeasibility(),
            'suggestions': self._generate_feasibility_suggestions()
        }

        # Save as text file
        with open('outputs/diagnostic_report.txt', 'w') as f:
            f.write("SCHEDULING PROBLEM DIAGNOSTIC REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Status: {diagnostic_data['status']}\n")
            f.write(f"Error: {diagnostic_data['error']}\n\n")

            f.write("PROBLEM ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            for analysis in diagnostic_data['problem_analysis']:
                f.write(f"- {analysis}\n")

            f.write("\nSUGGESTIONS:\n")
            f.write("-" * 20 + "\n")
            for suggestion in diagnostic_data['suggestions']:
                f.write(f"- {suggestion}\n")

    def _analyze_infeasibility(self) -> List[str]:
        """Analyze potential causes of infeasibility."""
        analysis = []

        # Check demand vs capacity
        total_demand = self.solver.demand['required_staff'].sum()
        total_employees = len(self.solver.employees)
        analysis.append(f"Total demand: {total_demand} staff positions")
        analysis.append(f"Total employees: {total_employees}")

        # Check availability coverage
        availability_coverage = {}
        for _, demand_row in self.solver.demand.iterrows():
            key = (demand_row['shift_id'], demand_row['date'], demand_row['location_id'])
            available_count = len(self.solver.availability[
                (self.solver.availability['shift_id'] == demand_row['shift_id']) &
                (self.solver.availability['date'] == demand_row['date']) &
                (self.solver.availability['available'] == 1)
            ])
            required = demand_row['required_staff']

            if available_count < required:
                analysis.append(f"Understaffed: {key} needs {required}, only {available_count} available")

        # Check for impossible constraints
        if self.solver.availability['available'].sum() == 0:
            analysis.append("ERROR: No employees marked as available for any shifts")

        # Check date consistency
        demand_dates = set(self.solver.demand['date'])
        avail_dates = set(self.solver.availability['date'])
        missing_dates = demand_dates - avail_dates
        if missing_dates:
            analysis.append(f"Missing availability data for dates: {missing_dates}")

        return analysis

    def _generate_feasibility_suggestions(self) -> List[str]:
        """Generate suggestions to make problem feasible."""
        suggestions = [
            "Increase employee availability (set more 'available'=1 in availability sheet)",
            "Reduce demand requirements (lower 'required_staff' in demand sheet)",
            "Add more employees to the employees sheet",
            "Extend the time limit for optimization",
            "Check that all shift_ids in demand exist in shifts sheet",
            "Check that all employee_ids in availability exist in employees sheet",
            "Ensure date formats are consistent across all sheets (YYYY-MM-DD)",
            "Verify that demand dates have corresponding availability data"
        ]
        return suggestions

    def generate_schedule_excel(self):
        """Generate final schedule Excel file."""
        if 'assignments' not in self.results:
            return

        assignments = pd.DataFrame(self.results['assignments'])

        # Add is_new_assignment column
        existing_set = set()
        if not self.solver.existing_schedule.empty:
            for _, row in self.solver.existing_schedule.iterrows():
                try:
                    date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                    existing_set.add((row['employee_id'], date_str,
                                    row['location_id'], row['shift_id']))
                except Exception:
                    continue

        if not assignments.empty:
            def check_new_assignment(row):
                try:
                    date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                    return 0 if (row['employee_id'], date_str,
                               row['location_id'], row['shift_id']) in existing_set else 1
                except Exception:
                    return 1

            assignments['is_new_assignment'] = assignments.apply(check_new_assignment, axis=1)

        assignments.to_excel('outputs/final_schedule.xlsx', index=False)

    def generate_payroll_excel(self):
        """Generate payroll summary Excel file."""
        if 'assignments' not in self.results or not self.results['assignments']:
            return

        assignments_df = pd.DataFrame(self.results['assignments'])

        payroll_data = []
        for emp_id in assignments_df['employee_id'].unique():
            emp_assignments = assignments_df[assignments_df['employee_id'] == emp_id]
            emp_data = self.solver.employees[self.solver.employees['employee_id'] == emp_id].iloc[0]

            total_hours = emp_assignments['hours'].sum()
            regular_hours = min(total_hours, emp_data['contract_max_hours'])
            overtime_hours = max(0, total_hours - emp_data['contract_max_hours'])

            gross_pay = (regular_hours * emp_data['hourly_rate'] +
                        overtime_hours * emp_data['hourly_rate'] * emp_data['overtime_multiplier'])

            payroll_data.append({
                'employee_id': emp_id,
                'total_hours': total_hours,
                'regular_hours': regular_hours,
                'overtime_hours': overtime_hours,
                'gross_pay': gross_pay
            })

        payroll_df = pd.DataFrame(payroll_data)
        payroll_df.to_excel('outputs/payroll_summary.xlsx', index=False)

    def generate_changelog_excel(self):
        """Generate changelog Excel file."""
        changelog_data = []

        # Compare with existing schedule
        existing_assignments = set()
        if not self.solver.existing_schedule.empty:
            for _, row in self.solver.existing_schedule.iterrows():
                existing_assignments.add((row['employee_id'], row['date'],
                                        row['location_id'], row['shift_id']))

        new_assignments = set()
        if 'assignments' in self.results:
            for assignment in self.results['assignments']:
                new_assignments.add((assignment['employee_id'], assignment['date'],
                                   assignment['location_id'], assignment['shift_id']))

        # Removed assignments
        for old_assignment in existing_assignments - new_assignments:
            emp_id, date, loc_id, shift_id = old_assignment
            changelog_data.append({
                'employee_id': emp_id,
                'date': date,
                'location_id': loc_id,
                'shift_id': shift_id,
                'old_assignment': f"{shift_id}@{loc_id}",
                'new_assignment': 'REMOVED',
                'reason': 'Optimization',
                'change_cost': self.solver.config.change_penalty
            })

        # Added assignments
        for new_assignment in new_assignments - existing_assignments:
            emp_id, date, loc_id, shift_id = new_assignment
            changelog_data.append({
                'employee_id': emp_id,
                'date': date,
                'location_id': loc_id,
                'shift_id': shift_id,
                'old_assignment': 'NONE',
                'new_assignment': f"{shift_id}@{loc_id}",
                'reason': 'Optimization',
                'change_cost': self.solver.config.change_penalty
            })

        if changelog_data:
            changelog_df = pd.DataFrame(changelog_data)
            changelog_df.to_excel('outputs/changelog.xlsx', index=False)

    def generate_insights_json(self):
        """Generate insights JSON file."""
        insights = {
            'timestamp': self.timestamp,
            'optimization_status': self.results.get('status', 'UNKNOWN'),
            'solve_time_seconds': self.results.get('solve_time', 0),
            'objective_value': self.results.get('objective_value', 0),
            'kpis': self._calculate_kpis(),
            'cost_breakdown': self._calculate_cost_breakdown(),
            'coverage_analysis': self._analyze_coverage(),
            'suggestions': self._generate_suggestions()
        }

        with open('outputs/insights.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)

    def generate_aggregated_results(self) -> pd.DataFrame:
        """Generate aggregated results report table."""
        if 'assignments' not in self.results or not self.results['assignments']:
            return pd.DataFrame()

        aggregated_data = []

        # Get all demand rows
        for _, demand_row in self.solver.demand.iterrows():
            date = demand_row['date']
            location_id = demand_row['location_id']
            shift_id = demand_row['shift_id']
            required_staff = int(demand_row['required_staff'])

            # Count actual assignments for this demand
            count_of_employees = 0
            for assignment in self.results['assignments']:
                if (assignment['shift_id'] == shift_id and
                    pd.to_datetime(assignment['date']).date() == date.date() and
                    assignment['location_id'] == location_id):
                    count_of_employees += 1

            # Calculate difference
            difference = count_of_employees - required_staff

            # Format difference with proper sign
            if difference > 0:
                difference_str = f"+{difference}"
            elif difference < 0:
                difference_str = f"{difference}"
            else:
                difference_str = "0"

            # Determine state
            if difference < 0:
                state = "Understaffed"
                state_display = "❌Understaffed"
            elif difference > 0:
                state = "Overstaffed"
                state_display = "⚠️Overstaffed"
            else:
                state = "Fulfilled"
                state_display = "✅Fulfilled"

            aggregated_data.append({
                'date': date.strftime('%m/%d/%Y'),
                'location_id': location_id,
                'shift_id': shift_id,
                'Count of employee_id': count_of_employees,
                'required_staff': required_staff,
                'difference': difference_str,
                'state': state_display,
                'state_raw': state  # Keep original state for metrics calculations
            })


        # Sort by date, then by location, then by shift
        aggregated_df = pd.DataFrame(aggregated_data)
        if not aggregated_df.empty:
            aggregated_df['date'] = pd.to_datetime(aggregated_df['date'])
            aggregated_df = aggregated_df.sort_values(['date', 'location_id', 'shift_id'])
            aggregated_df['date'] = aggregated_df['date'].dt.strftime('%m/%d/%Y')

        return aggregated_df

    def generate_aggregated_results_file(self):
        """Generate and save aggregated results to Excel file."""
        aggregated_df = self.generate_aggregated_results()
        if not aggregated_df.empty:
            aggregated_df.to_excel('outputs/aggregated_results.xlsx', index=False)

    def generate_charts(self):
        """Generate visualization charts."""
        try:
            self._generate_coverage_chart()
            self._generate_cost_breakdown_chart()
            self._generate_understaffed_chart()
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")

    def _calculate_kpis(self) -> Dict[str, Any]:
        """Calculate key performance indicators."""
        # Use the same calculation logic as dashboard for consistency
        try:
            metrics = calculate_dashboard_metrics(self.results, self.solver)
        except Exception as e:
            # Fallback to basic metrics if calculation fails
            logger.warning(f"Error calculating dashboard metrics: {e}")
            metrics = {
                'coverage_percentage': 0.0,
                'total_cost': 0.0,
                'overtime_hours': 0.0,
                'overtime_cost': 0.0,
                'total_changes': 0,
                'understaffed_shifts': 0,
                'overstaffed_shifts': 0,
                'total_assignments': 0,
                'total_shortage': 0,
                'total_excess': 0
            }

        total_demand = self.solver.demand['required_staff'].sum() if self.solver.demand is not None and not self.solver.demand.empty else 0

        return {
            'coverage_percentage': round(metrics['coverage_percentage'], 2),
            'total_assignments': metrics['total_assignments'],
            'understaffed_shifts': metrics['understaffed_shifts'],
            'overstaffed_shifts': metrics['overstaffed_shifts'],
            'total_shortage': metrics['total_shortage'],
            'total_excess': metrics['total_excess'],
            'total_demand': total_demand,
            'overtime_hours': metrics['overtime_hours'],
            'overtime_cost': metrics['overtime_cost'],
            'total_cost': metrics['total_cost'],
            'total_changes': metrics['total_changes']
        }

    def _calculate_cost_breakdown(self) -> Dict[str, float]:
        """Calculate detailed cost breakdown."""
        if not self.results.get('assignments'):
            return {'total_cost': 0, 'labor_cost': 0, 'overtime_cost': 0}

        assignments_df = pd.DataFrame(self.results['assignments'])
        total_labor_cost = 0
        total_overtime_cost = 0

        for _, assignment in assignments_df.iterrows():
            emp_data = self.solver.employees[
                self.solver.employees['employee_id'] == assignment['employee_id']].iloc[0]
            labor_cost = assignment['hours'] * emp_data['hourly_rate']
            total_labor_cost += labor_cost

        # Add overtime costs
        for emp_id, ot_hours in self.results.get('overtime', {}).items():
            emp_data = self.solver.employees[self.solver.employees['employee_id'] == emp_id].iloc[0]
            ot_cost = ot_hours * emp_data['hourly_rate'] * emp_data['overtime_multiplier']
            total_overtime_cost += ot_cost

        return {
            'total_cost': total_labor_cost + total_overtime_cost,
            'labor_cost': total_labor_cost,
            'overtime_cost': total_overtime_cost
        }

    def _analyze_coverage(self) -> Dict[str, Any]:
        """Analyze coverage by location and shift."""
        coverage_data = []

        for _, demand_row in self.solver.demand.iterrows():
            shift_id = demand_row['shift_id']
            date = demand_row['date']
            location_id = demand_row['location_id']
            required = demand_row['required_staff']

            # Count assignments for this demand
            assignments = self.results.get('assignments', [])
            assigned = sum(1 for a in assignments
                          if a['shift_id'] == shift_id and
                             pd.to_datetime(a['date']).date() == date.date() and
                             a['location_id'] == location_id)

            shortage = max(0, required - assigned)
            coverage_pct = (assigned / required * 100) if required > 0 else 100

            coverage_data.append({
                'location_id': location_id,
                'shift_id': shift_id,
                'date': date.strftime('%Y-%m-%d'),
                'required': required,
                'assigned': assigned,
                'shortage': shortage,
                'coverage_percentage': coverage_pct
            })

        return {'coverage_details': coverage_data}

    def create_gantt_chart(self, assignments_df, solver, filter_employee=None, filter_location=None):
            """Create Gantt chart for employee assignments."""
            if assignments_df.empty:
                return None

            # Filter data if needed
            filtered_df = assignments_df.copy()
            if filter_employee and filter_employee != "All":
                filtered_df = filtered_df[filtered_df['employee_id'] == filter_employee]
            if filter_location and filter_location != "All":
                filtered_df = filtered_df[filtered_df['location_id'] == filter_location]

            if filtered_df.empty:
                return None

            # Prepare data for Gantt chart
            gantt_data = []

            # Get employee names mapping
            emp_names = {}
            if solver and not solver.employees.empty:
                emp_names = solver.employees.set_index('employee_id')['name'].to_dict()

            # Create time slots (day-shift combinations)
            dates = sorted(filtered_df['date'].unique())
            shifts = sorted(filtered_df['shift_id'].unique())

            # Create a mapping of shift times
            shift_times = {}
            if solver and not solver.shifts.empty:
                for _, shift in solver.shifts.iterrows():
                    shift_times[shift['shift_id']] = {
                        'start': shift['start_time'],
                        'end': shift['end_time']
                    }

            for _, row in filtered_df.iterrows():
                emp_name = emp_names.get(row['employee_id'], row['employee_id'])

                # Create datetime for start and end
                date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                shift_info = shift_times.get(row['shift_id'], {'start': '09:00', 'end': '17:00'})

                start_datetime = f"{date_str} {shift_info['start']}"
                end_datetime = f"{date_str} {shift_info['end']}"

                gantt_data.append({
                    'Task': emp_name,
                    'Start': start_datetime,
                    'Finish': end_datetime,
                    'Resource': f"{row['shift_id']} @ {row['location_id']}",
                    'Description': f"Employee: {emp_name}<br>Date: {date_str}<br>Shift: {row['shift_id']}<br>Location: {row['location_id']}<br>Hours: {row['hours']}"
                })

            if not gantt_data:
                return None

            # Create Gantt chart
            try:
                fig = ff.create_gantt(
                    gantt_data,
                    colors=['#FF6B6B', '#AFFFAF', '#45B7D1', '#FFAE5D', '#FFEAA7', '#DDA0DD', '#B9E5D9'],
                    index_col='Resource',
                    show_colorbar=True,
                    group_tasks=True,
                    showgrid_x=True,
                    showgrid_y=True,
                    title="Employee Schedule Gantt Chart"
                )

                fig.update_layout(
                    height=max(400, len(set(gantt_data[0]['Task'] for gantt_data in [gantt_data])) * 30),
                    xaxis_title="Time",
                    yaxis_title="Employees",
                    hovermode='closest'
                )

                return fig

            except Exception as e:
                # Fallback to simple bar chart if Gantt fails
                import plotly.express as px

                # Create a simple timeline chart as fallback
                gantt_df = pd.DataFrame(gantt_data)
                gantt_df['Start'] = pd.to_datetime(gantt_df['Start'])
                gantt_df['Finish'] = pd.to_datetime(gantt_df['Finish'])

                fig = px.timeline(
                    gantt_df,
                    x_start="Start",
                    x_end="Finish",
                    y="Task",
                    color="Resource",
                    hover_data=["Description"],
                    title="Employee Schedule Timeline"
                )

                fig.update_layout(
                    height=max(400, gantt_df['Task'].nunique() * 40),
                    yaxis={'categoryorder': 'category ascending'}
                )

                return fig

    def _generate_suggestions(self) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []

        understaffed_shifts = len(self.results.get('understaffing', []))
        if understaffed_shifts > 0:
            suggestions.append(f"Consider hiring more staff - {understaffed_shifts} shifts are understaffed")

        total_changes = len(self.results.get('changes', []))
        if total_changes > 10:
            suggestions.append("High number of schedule changes - consider reducing change penalties")

        if self.results.get('status') == 'TIMEOUT':
            suggestions.append("Optimization timed out - consider increasing time limit or simplifying constraints")

        return suggestions

    def _generate_coverage_chart(self):
        """Generate coverage by store chart."""
        if not self.results.get('assignments'):
            return

        coverage_analysis = self._analyze_coverage()
        coverage_df = pd.DataFrame(coverage_analysis['coverage_details'])

        # Aggregate by location
        location_coverage = coverage_df.groupby('location_id').agg({
            'coverage_percentage': 'mean',
            'shortage': 'sum'
        }).reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Coverage percentage by location
        ax1.bar(location_coverage['location_id'], location_coverage['coverage_percentage'])
        ax1.set_title('Average Coverage Percentage by Location')
        ax1.set_ylabel('Coverage %')
        ax1.set_ylim(0, 100)

        # Total shortage by location
        ax2.bar(location_coverage['location_id'], location_coverage['shortage'])
        ax2.set_title('Total Staff Shortage by Location')
        ax2.set_ylabel('Shortage (Staff Hours)')

        plt.tight_layout()
        plt.savefig('outputs/charts/coverage_by_store.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_cost_breakdown_chart(self):
        """Generate cost breakdown pie chart."""
        cost_breakdown = self._calculate_cost_breakdown()

        labels = ['Labor Cost', 'Overtime Cost']
        sizes = [cost_breakdown['labor_cost'], cost_breakdown['overtime_cost']]

        # Remove zero values
        non_zero = [(label, size) for label, size in zip(labels, sizes) if size > 0]
        if not non_zero:
            return

        labels, sizes = zip(*non_zero)

        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Cost Breakdown')
        plt.axis('equal')

        plt.savefig('outputs/charts/cost_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_understaffed_chart(self):
        """Generate top understaffed shifts chart."""
        understaffing = self.results.get('understaffing', [])
        if not understaffing:
            return

        # Sort by shortage and take top 10
        understaffing_sorted = sorted(understaffing, key=lambda x: x['shortage'], reverse=True)[:10]

        if not understaffing_sorted:
            return

        locations = [f"{u['location_id']}-{u['shift_id']}" for u in understaffing_sorted]
        shortages = [u['shortage'] for u in understaffing_sorted]

        plt.figure(figsize=(12, 6))
        plt.barh(locations, shortages)
        plt.title('Top Understaffed Shifts')
        plt.xlabel('Staff Shortage')
        plt.tight_layout()

        plt.savefig('outputs/charts/top_understaffed.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_streamlit_app():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Dynamic Staff Scheduling Engine",
        page_icon="📅",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'solver' not in st.session_state:
        st.session_state.solver = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'sample_generated' not in st.session_state:
        st.session_state.sample_generated = False
    if 'previous_metrics' not in st.session_state:
        st.session_state.previous_metrics = None
    if 'dashboard_metrics' not in st.session_state:
        st.session_state.dashboard_metrics = None

    # Sidebar configuration
    st.sidebar.markdown(
    """
    <style>
    .custom-title {
        font-size: 40px;
        font-weight: 900;
        color: white;
        background-color: black;
        padding: 10px 20px;
        border-radius: 10px;
        display: inline-block;
    }
    .custom-title span {
        color: #F36B21; /* orange for the 'B' */
    }
    </style>
    <div class="custom-title">2<span>B</span></div>
    """,
    unsafe_allow_html=True
)


    # Configuration section
    st.sidebar.header("Optimization Settings")
    time_limit = st.sidebar.slider("Time Limit (seconds)", 30, 3600, 600)  # Increased max to 3600 (1 hour)
    understaff_penalty = st.sidebar.number_input("Understaffing Penalty", 100.0, 10000.0, 1000.0)
    overstaff_penalty = st.sidebar.number_input("Overstaffing Penalty", 10.0, 1000.0, 100.0)
    change_penalty = st.sidebar.number_input("Change Penalty", 0.0, 100.0, 50.0)
    fairness_weight = st.sidebar.number_input("Fairness Weight", 0.0, 50.0, 10.0)
    solver_type = st.sidebar.selectbox("Solver", ["CP-SAT", "PULP"])

    config = SchedulingConfig(
        time_limit_seconds=time_limit,
        understaff_penalty=understaff_penalty,
        overstaff_penalty=overstaff_penalty,
        change_penalty=change_penalty,
        fairness_weight=fairness_weight,
        solver_type=solver_type
    )

    # Main content
    st.title("Dynamic Staffing Engine")
    st.markdown("### Organize. Optimize. Achieve.")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard", "Data Import", "Calendar", "Scenario Manager", "Results"
    ])

    with tab1:
        show_dashboard_tab(config)

    with tab2:
        show_data_import_tab(config)

    with tab3:
        show_calendar_tab()

    with tab4:
        show_scenario_manager_tab(config)

    with tab5:
        show_results_tab()

def calculate_dashboard_metrics(results, solver):
    """Calculate dynamic dashboard metrics from optimization results."""
    metrics = {
        'coverage_percentage': 0.0,
        'total_cost': 0.0,
        'overtime_hours': 0.0,
        'overtime_cost': 0.0,
        'total_changes': 0,
        'understaffed_shifts': 0,
        'overstaffed_shifts': 0,
        'total_assignments': 0,
        'total_shortage': 0,
        'total_excess': 0
    }

    # Safety checks
    try:
        if not results:
            return metrics

        if results.get('status') not in ['OPTIMAL', 'FEASIBLE']:
            return metrics

        if not solver:
            return metrics

        # Additional safety check for solver attributes
        if not hasattr(solver, 'demand') or solver.demand is None:
            return metrics

        if not hasattr(solver, 'employees') or solver.employees is None:
            return metrics

    except Exception as e:
        logger.warning(f"Error in safety checks for calculate_dashboard_metrics: {e}")
        return metrics
    try:
        # Calculate total assignments
        assignments = results.get('assignments', [])
        metrics['total_assignments'] = len(assignments)

        # Calculate coverage percentage
        if solver and solver.demand is not None and not solver.demand.empty:
            total_demand = solver.demand['required_staff'].sum()
            total_shortage = sum(u['shortage'] for u in results.get('understaffing', []))
            if total_demand > 0:
                metrics['coverage_percentage'] = ((total_demand - total_shortage) / total_demand) * 100
            else:
                metrics['coverage_percentage'] = 100.0

        # Calculate labor costs
        if assignments and solver and solver.employees is not None:
            total_labor_cost = 0.0

            for assignment in assignments:
                emp_data = solver.employees[solver.employees['employee_id'] == assignment['employee_id']]
                if not emp_data.empty:
                    hourly_rate = emp_data.iloc[0]['hourly_rate']
                    hours = assignment.get('hours', 0)
                    total_labor_cost += hours * hourly_rate

            metrics['total_cost'] = total_labor_cost

        # Calculate overtime
        overtime_dict = results.get('overtime', {})
        if overtime_dict and solver and solver.employees is not None:
            total_overtime_hours = sum(overtime_dict.values())
            total_overtime_cost = 0.0

            for emp_id, ot_hours in overtime_dict.items():
                emp_data = solver.employees[solver.employees['employee_id'] == emp_id]
                if not emp_data.empty:
                    emp_info = emp_data.iloc[0]
                    hourly_rate = emp_info['hourly_rate']
                    overtime_multiplier = emp_info.get('overtime_multiplier', 1.5)
                    total_overtime_cost += ot_hours * hourly_rate * overtime_multiplier

            metrics['overtime_hours'] = total_overtime_hours
            metrics['overtime_cost'] = total_overtime_cost
            metrics['total_cost'] += total_overtime_cost

        # Calculate understaffing and overstaffing
        understaffing = results.get('understaffing', [])
        overstaffing = results.get('overstaffing', [])

        metrics['understaffed_shifts'] = len(understaffing)
        metrics['overstaffed_shifts'] = len(overstaffing)
        metrics['total_shortage'] = sum(u['shortage'] for u in understaffing)
        metrics['total_excess'] = sum(o['excess'] for o in overstaffing)

        # Calculate changes
        metrics['total_changes'] = len(results.get('changes', []))

    except Exception as e:
        logger.warning(f"Error calculating dashboard metrics: {e}")
        # Return partially calculated metrics

    return metrics

def show_dashboard_tab(config):
    """Dashboard tab with KPIs and controls."""
    st.header("📊 Executive Dashboard")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.optimization_results:
            results = st.session_state.optimization_results
            try:
                metrics = calculate_dashboard_metrics(results, st.session_state.solver)
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")
                # Use fallback metrics
                metrics = {
                    'coverage_percentage': 0.0,
                    'total_cost': 0.0,
                    'overtime_hours': 0.0,
                    'overtime_cost': 0.0,
                    'total_changes': 0,
                    'understaffed_shifts': 0,
                    'overstaffed_shifts': 0,
                    'total_assignments': len(results.get('assignments', [])),
                    'total_shortage': 0,
                    'total_excess': 0
                }

            # KPI Cards
            col1a, col1b, col1c, col1d = st.columns(4)

            with col1a:
                coverage = metrics['coverage_percentage']
                # Calculate delta from previous run if available
                coverage_delta = None
                if hasattr(st.session_state, 'previous_metrics') and st.session_state.previous_metrics is not None:
                    prev_coverage = st.session_state.previous_metrics.get('coverage_percentage', coverage)
                    coverage_delta = f"{coverage - prev_coverage:+.1f}%"
                st.metric("Coverage", f"{coverage:.1f}%", coverage_delta)

            with col1b:
                total_cost = metrics['total_cost']
                cost_delta = None
                if hasattr(st.session_state, 'previous_metrics') and st.session_state.previous_metrics is not None:
                    prev_cost = st.session_state.previous_metrics.get('total_cost', total_cost)
                    cost_delta = f"{total_cost - prev_cost:+,.0f}"
                st.metric("Total Cost", f"EGP{total_cost:,.0f}", cost_delta)

            with col1c:
                overtime_hours = metrics['overtime_hours']
                ot_delta = None
                if hasattr(st.session_state, 'previous_metrics') and st.session_state.previous_metrics is not None:
                    prev_ot = st.session_state.previous_metrics.get('overtime_hours', overtime_hours)
                    ot_delta = f"{overtime_hours - prev_ot:+.0f}"
                st.metric("Overtime Hours", f"{overtime_hours:.0f}", ot_delta)

            with col1d:
                total_changes = metrics['total_changes']
                changes_delta = None
                if hasattr(st.session_state, 'previous_metrics') and st.session_state.previous_metrics is not None:
                    prev_changes = st.session_state.previous_metrics.get('total_changes', total_changes)
                    changes_delta = f"{total_changes - prev_changes:+d}"
                st.metric("Changes", f"{total_changes}", changes_delta)

            # Store current metrics for next comparison
            st.session_state.previous_metrics = metrics

            # Status indicators
            st.subheader("🚦 System Status")

            status_col1, status_col2, status_col3 = st.columns(3)

            with status_col1:
                status = results.get('status', 'UNKNOWN')
                color = {"OPTIMAL": "🟢", "FEASIBLE": "🟡", "INFEASIBLE": "🔴"}.get(status, "⚪")
                st.write(f"{color} Solver Status: **{status}**")

            with status_col2:
                solve_time = results.get('solve_time', 0) or 0
                st.write(f"⏱️ Solve Time: **{solve_time:.2f}s**")

            with status_col3:
                # Use calculated total cost instead of objective value
                total_cost = metrics['total_cost']
                st.write(f"💰 Total Cost: **EGP{total_cost:,.0f}**")

            # Add additional status row
            st.markdown("---")
            add_status_col1, add_status_col2, add_status_col3, add_status_col4 = st.columns(4)

            with add_status_col1:
                understaffed = metrics['understaffed_shifts']
                color = "🔴" if understaffed > 0 else "🟢"
                st.write(f"{color} Understaffed: **{understaffed}**")

            with add_status_col2:
                overstaffed = metrics['overstaffed_shifts']
                color = "🟡" if overstaffed > 0 else "🟢"
                st.write(f"{color} Overstaffed: **{overstaffed}**")

            with add_status_col3:
                assignments = metrics['total_assignments']
                st.write(f"📋 Assignments: **{assignments}**")

            with add_status_col4:
                ot_cost = metrics['overtime_cost']
                st.write(f"💰 OT Cost: **EGP{ot_cost:,.0f}**")

            # Enhanced Dashboard Summary
            st.markdown("---")
            st.subheader("📊 Performance Summary")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.markdown("**📈 Staffing Analysis**")
                # Safe access to solver data
                try:
                    if st.session_state.solver and hasattr(st.session_state.solver, 'demand') and st.session_state.solver.demand is not None:
                        total_demand = st.session_state.solver.demand['required_staff'].sum()
                    else:
                        total_demand = 0
                except Exception as e:
                    total_demand = 0

                st.write(f"• Total Demand: {total_demand} positions")
                st.write(f"• Total Assigned: {metrics['total_assignments']} positions")
                st.write(f"• Staff Shortage: {metrics['total_shortage']} positions")
                st.write(f"• Staff Excess: {metrics['total_excess']} positions")

            with summary_col2:
                st.markdown("**💰 Cost Analysis**")
                regular_cost = metrics['total_cost'] - metrics['overtime_cost']
                st.write(f"• Regular Labor: EGP{regular_cost:,.0f}")
                st.write(f"• Overtime Cost: EGP{metrics['overtime_cost']:,.0f}")
                st.write(f"• Total Labor Cost: EGP{metrics['total_cost']:,.0f}")
                if metrics['total_cost'] > 0:
                    ot_percentage = (metrics['overtime_cost'] / metrics['total_cost']) * 100
                    st.write(f"• Overtime %: {ot_percentage:.1f}%")
        else:
            st.info("📋 No optimization results yet. Load data and run optimization to see KPIs.")

    with col2:
        st.subheader("🎯 Quick Actions")

        if st.button("🔄 Run Optimization", key="run_optimization_btn", type="primary", use_container_width=True):
            if st.session_state.solver and st.session_state.data_loaded:
                with st.spinner("🔧 Optimizing schedule..."):
                    try:
                        st.session_state.solver.config = config
                        results = st.session_state.solver.optimize()
                        st.session_state.optimization_results = results

                        if results['status'] in ['OPTIMAL', 'FEASIBLE']:
                            # Generate reports
                            report_gen = ReportGenerator(st.session_state.solver, results)
                            report_gen.generate_all_outputs()
                            # Calculate and display summary metrics
                            try:
                                metrics = calculate_dashboard_metrics(results, st.session_state.solver)
                                st.success(f"✅ Optimization completed! Coverage: {metrics['coverage_percentage']:.1f}%, Cost: EGP{metrics['total_cost']:,.0f}")
                            except Exception as e:
                                st.warning(f"Metrics calculation error: {str(e)}")
                                st.success("✅ Optimization completed successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Optimization failed: {results.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Error during optimization: {str(e)}")
            else:
                st.warning("⚠️ Please load data first!")

        if st.button("📊 Generate Sample Data", key="generate_sample_btn", use_container_width=True):
            try:
                filename = SampleDataGenerator.save_sample_data()
                st.session_state.sample_generated = True
                st.success(f"✅ Sample data generated: {filename}")
            except Exception as e:
                st.error(f"❌ Error generating sample data: {str(e)}")

        if st.button("📤 Publish Schedule", key="publish_schedule_btn", use_container_width=True):
            if st.session_state.optimization_results:
                try:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    published_file = f'outputs/existing_schedule_published_{timestamp}.xlsx'
                    
                    if st.session_state.optimization_results.get('assignments'):
                        assignments_df = pd.DataFrame(st.session_state.optimization_results['assignments'])
                        assignments_df.to_excel(published_file, index=False)
                        
                        # Store the published file path in session state
                        st.session_state.published_file = published_file
                        st.session_state.published_timestamp = timestamp
                        
                        st.success(f"✅ Schedule published: {published_file}")
                        st.rerun()
                    else:
                        st.warning("⚠️ No assignments to publish")
                except Exception as e:
                    st.error(f"❌ Error publishing schedule: {str(e)}")
            else:
                st.warning("⚠️ No optimization results to publish!")

        # Download button for published schedule (appears after publishing)
        if hasattr(st.session_state, 'published_file') and st.session_state.published_file:
            try:
                import os
                if os.path.exists(st.session_state.published_file):
                    with open(st.session_state.published_file, 'rb') as file:
                        st.download_button(
                            label="⬇️ Download Published Schedule",
                            data=file,
                            file_name=f"schedule_published_{st.session_state.published_timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_published_schedule_btn",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"❌ Error preparing download: {str(e)}")

        if st.button("🧪 Test Run", key="test_run_btn", use_container_width=True):
            try:
                # Generate and load sample data
                filename = SampleDataGenerator.save_sample_data()

                # Initialize solver with sample data
                solver = SchedulingSolver(config)
                solver.load_data(filename)
                st.session_state.solver = solver
                st.session_state.data_loaded = True

                # Run optimization
                with st.spinner("🧪 Running test optimization..."):
                    results = solver.optimize()
                    st.session_state.optimization_results = results

                    if results['status'] in ['OPTIMAL', 'FEASIBLE']:
                        report_gen = ReportGenerator(solver, results)
                        report_gen.generate_all_outputs()
                        st.success("✅ Test run completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Test run failed: {results.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Error during test run: {str(e)}")

def show_data_import_tab(config):
    """Data import and validation tab."""
    st.header("📊 Data Import & Validation")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Upload Data")

        upload_mode = st.radio(
            "Choose input mode:",
            ["Upload Excel File", "Use Sample Dataset", "Load from Local File"]
        )

        if upload_mode == "Upload Excel File":
            uploaded_file = st.file_uploader(
                "Upload input_data.xlsx",
                type=['xlsx'],
                help="Upload Excel workbook with required sheets"
            )

            if uploaded_file is not None:
                # Save uploaded file temporarily with unique name
                import time
                temp_path = f"temp_input_data_{int(time.time())}.xlsx"

                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Validate and load
                    validation_results = DataValidator.validate_workbook(temp_path)

                    if validation_results['valid']:
                        try:
                            solver = SchedulingSolver(config)
                            solver.load_data(temp_path)
                            st.session_state.solver = solver
                            st.session_state.data_loaded = True
                            st.success("✅ Data loaded successfully!")
                        except Exception as e:
                            st.error(f"❌ Error loading data: {str(e)}")
                    else:
                        st.error("❌ Data validation failed!")

                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")

                finally:

                    # Clean up temp file with improved retry mechanism
                    def cleanup_temp_file(temp_path):
                        """Clean up temporary file with proper error handling."""
                        if not os.path.exists(temp_path):
                            return True

                        import time
                        import gc

                        # Force garbage collection to release file handles
                        gc.collect()

                        for attempt in range(10):  # Increased attempts
                            try:
                                # Try to close any open file handles
                                try:
                                    import psutil
                                    current_process = psutil.Process()
                                    for file_handle in current_process.open_files():
                                        if temp_path in file_handle.path:
                                            file_handle.close()
                                except:
                                    pass  # psutil might not be available

                                os.remove(temp_path)
                                return True
                            except PermissionError:
                                if attempt < 9:  # Not the last attempt
                                    time.sleep(1.0)  # Longer wait time
                                else:
                                    # Final attempt - try renaming first
                                    try:
                                        import uuid
                                        backup_name = f"{temp_path}.backup_{uuid.uuid4().hex[:8]}"
                                        os.rename(temp_path, backup_name)
                                        # Schedule for cleanup later
                                        import atexit
                                        atexit.register(lambda: os.remove(backup_name) if os.path.exists(backup_name) else None)
                                        return True
                                    except:
                                        return False
                            except FileNotFoundError:
                                return True  # Already deleted
                            except Exception:
                                if attempt == 9:  # Last attempt
                                    return False
                                time.sleep(0.5)

                        return False

                    # Use the improved cleanup function
                    try:
                        if cleanup_temp_file(temp_path):
                            pass  # Successfully cleaned up
                        else:
                            # Only show warning if cleanup truly failed
                            pass  # Silently ignore - file will be cleaned up eventually
                    except Exception:
                        pass  # Ignore any cleanup errors



        elif upload_mode == "Use Sample Dataset":
            if st.button("🎯 Load Sample Data", type="primary"):
                try:
                    filename = SampleDataGenerator.save_sample_data()
                    solver = SchedulingSolver(config)
                    solver.load_data(filename)
                    st.session_state.solver = solver
                    st.session_state.data_loaded = True
                    st.success("✅ Sample data loaded!")
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")

        else:  # Load from Local File
            local_file = "input_data.xlsx"
            if st.button("📂 Load Local File", type="primary"):
                if os.path.exists(local_file):
                    validation_results = DataValidator.validate_workbook(local_file)

                    if validation_results['valid']:
                        try:
                            solver = SchedulingSolver(config)
                            solver.load_data(local_file)
                            st.session_state.solver = solver
                            st.session_state.data_loaded = True
                            st.success("✅ Local data loaded successfully!")
                        except Exception as e:
                            st.error(f"❌ Error loading data: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
                    else:
                        st.error("❌ Local file validation failed!")
                        for error in validation_results['errors']:
                            st.write(f"- {error}")
                        for warning in validation_results['warnings']:
                            st.write(f"⚠️ {warning}")
                else:
                    st.error("❌ File 'input_data.xlsx' not found in current directory!")

            # Add diagnostic button
            if st.button("🔍 Analyze File Structure"):
                if os.path.exists(local_file):
                    try:
                        # Read and analyze file
                        xl_file = pd.ExcelFile(local_file)
                        st.write(f"**Available sheets:** {xl_file.sheet_names}")

                        for sheet in xl_file.sheet_names:
                            try:
                                df = pd.read_excel(local_file, sheet_name=sheet)
                                st.write(f"**Sheet '{sheet}':** {df.shape[0]} rows, {df.shape[1]} columns")
                                st.write(f"Columns: {list(df.columns)}")

                                # Show sample data
                                if not df.empty:
                                    st.write("Sample data:")
                                    st.dataframe(df.head(3))

                                    # Check for issues
                                    if df.isnull().all().any():
                                        empty_cols = df.columns[df.isnull().all()].tolist()
                                        st.warning(f"Empty columns in {sheet}: {empty_cols}")

                                    if 'date' in df.columns:
                                        st.write(f"Date range: {df['date'].min()} to {df['date'].max()}")

                                st.write("---")
                            except Exception as e:
                                st.error(f"Error reading sheet {sheet}: {e}")
                    except Exception as e:
                        st.error(f"Error analyzing file: {e}")
                else:
                    st.error("File not found!")

    with col2:
        st.subheader("🔍 Validation Results")

        if st.session_state.data_loaded and st.session_state.solver:
            # Show data summary
            solver = st.session_state.solver

            st.success("✅ Data validation passed!")

            # Data summary
            with st.expander("📋 Data Summary", expanded=True):
                col2a, col2b, col2c = st.columns(3)

                with col2a:
                    st.metric("Employees", len(solver.employees))
                    st.metric("Shifts", len(solver.shifts))

                with col2b:
                    st.metric("Locations", solver.demand['location_id'].nunique())
                    st.metric("Days", solver.demand['date'].nunique())

                with col2c:
                    st.metric("Total Demand", solver.demand['required_staff'].sum())
                    st.metric("Existing Assignments", len(solver.existing_schedule))

            # Show sample data
            st.subheader("👥 Employees Preview")
            st.dataframe(solver.employees.head(), use_container_width=True)

            st.subheader("⏰ Shifts Preview")
            st.dataframe(solver.shifts, use_container_width=True)

            st.subheader("📈 Demand Preview")
            st.dataframe(solver.demand.head(), use_container_width=True)

        else:
            st.info("📋 Upload or load data to see validation results")

def show_calendar_tab():
    """Interactive calendar/roster tab."""
    st.header("📅 Schedule Calendar")

    if not st.session_state.data_loaded or not st.session_state.solver:
        st.warning("⚠️ Please load data first to view calendar")
        return

    solver = st.session_state.solver

    # Date range selector
    dates = sorted(solver.demand['date'].unique())
    if len(dates) > 0:
        selected_date = st.selectbox("📅 Select Date", dates)

        # Location selector
        locations = sorted(solver.demand['location_id'].unique())
        selected_location = st.selectbox("📍 Select Location", locations)

        # Filter data for selected date and location
        day_demand = solver.demand[
            (solver.demand['date'] == selected_date) &
            (solver.demand['location_id'] == selected_location)
        ]

        if len(day_demand) > 0:
            st.subheader(f"📊 Schedule for {selected_date.strftime('%Y-%m-%d')} at {selected_location}")

            # Create schedule grid
            schedule_data = []
            for _, demand_row in day_demand.iterrows():
                shift_id = demand_row['shift_id']
                required = demand_row['required_staff']

                # Get current assignments
                current_assignments = []
                if st.session_state.optimization_results and 'assignments' in st.session_state.optimization_results:
                    for assignment in st.session_state.optimization_results['assignments']:
                        if (pd.to_datetime(assignment['date']).date() == selected_date.date() and
                            assignment['location_id'] == selected_location and
                            assignment['shift_id'] == shift_id):
                            current_assignments.append(assignment['employee_id'])

                # Get shift details
                shift_info = solver.shifts[solver.shifts['shift_id'] == shift_id].iloc[0]

                schedule_data.append({
                    'Shift': shift_id,
                    'Time': f"{shift_info['start_time']} - {shift_info['end_time']}",
                    'Required': required,
                    'Assigned': len(current_assignments),
                    'Staff': ', '.join(current_assignments) if current_assignments else 'None',
                    'Status': '✅' if len(current_assignments) == required else '⚠️' if len(current_assignments) > required else '❌'
                })

            schedule_df = pd.DataFrame(schedule_data)
            st.dataframe(schedule_df, use_container_width=True)

            # Manual assignment interface
            st.subheader("✏️ Manual Assignment")

            col1, col2, col3 = st.columns(3)

            with col1:
                available_employees = list(solver.employees['employee_id'])
                selected_employee = st.selectbox("👤 Employee", available_employees)

            with col2:
                shifts_for_day = list(day_demand['shift_id'])
                selected_shift = st.selectbox("⏰ Shift", shifts_for_day)

            with col3:
                st.write("") # Spacing
                st.write("") # Spacing
                if st.button("➕ Assign"):
                    st.info(f"Manual assignment: {selected_employee} → {selected_shift}")
                    # In a full implementation, this would update the schedule
        else:
            st.info("📋 No demand data for selected date and location")
    else:
        st.info("📋 No date data available")

def show_scenario_manager_tab(config):
    """Scenario comparison and optimization tab."""
    st.header("🔍 Scenario Manager")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Optimization Parameters")

        # Parameter sliders
        time_limit = st.slider("⏱️ Time Limit (sec)", 30, 3600, config.time_limit_seconds)  # Increased max to 3600
        rho = st.slider("📊 Change Penalty", 1.0, 100.0, config.change_penalty)
        understaff_penalty = st.slider("❌ Understaffing Penalty", 100.0, 5000.0, config.understaff_penalty)
        overstaff_penalty = st.slider("⚠️ Overstaffing Penalty", 10.0, 1000.0, config.overstaff_penalty)
        fairness_weight = st.slider("⚖️ Fairness Weight", 0.0, 50.0, config.fairness_weight)

        # Update config
        scenario_config = SchedulingConfig(
            time_limit_seconds=int(time_limit),
            change_penalty=rho,
            understaff_penalty=understaff_penalty,
            overstaff_penalty=overstaff_penalty,
            fairness_weight=fairness_weight,
            solver_type=config.solver_type
        )

        if st.button("🚀 Run Scenario", type="primary"):
            if st.session_state.solver:
                with st.spinner("🔧 Running scenario optimization..."):
                    try:
                        st.session_state.solver.config = scenario_config
                        results = st.session_state.solver.optimize()
                        st.session_state.optimization_results = results

                        if results['status'] in ['OPTIMAL', 'FEASIBLE']:
                            report_gen = ReportGenerator(st.session_state.solver, results)
                            report_gen.generate_all_outputs()
                            st.success("✅ Scenario optimization completed!")
                            st.rerun()
                        else:
                            st.error(f"❌ Scenario failed: {results.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Error running scenario: {str(e)}")

    with col2:
        st.subheader("📊 Scenario Comparison")

        if st.session_state.optimization_results:
            results = st.session_state.optimization_results

            # Create comparison metrics
            col2a, col2b, col2c = st.columns(3)

            with col2a:
                st.metric(
                    "Status",
                    results.get('status', 'UNKNOWN'),
                    delta=None
                )

            with col2b:
                obj_value = results.get('objective_value', 0) or 0
                st.metric(
                    "Total Cost",
                    f"EGP{obj_value:,.0f}",
                    delta=None
                )

            with col2c:
                solve_time = results.get('solve_time', 0) or 0
                st.metric(
                    "Solve Time",
                    f"{solve_time:.2f}s",
                    delta=None
                )

            # Results visualization
            if results.get('assignments'):
                assignments_df = pd.DataFrame(results['assignments'])

                # Coverage analysis
                st.subheader("📈 Coverage Analysis")

                # Calculate coverage by shift
                coverage_by_shift = {}
                for _, demand_row in st.session_state.solver.demand.iterrows():
                    key = (demand_row['shift_id'], demand_row['date'], demand_row['location_id'])
                    required = demand_row['required_staff']

                    assigned = len([a for a in results['assignments']
                                  if (a['shift_id'], pd.to_datetime(a['date']).date(), a['location_id']) ==
                                     (key[0], key[1].date(), key[2])])

                    coverage_pct = (assigned / required * 100) if required > 0 else 100
                    coverage_by_shift[key] = coverage_pct

                if coverage_by_shift:
                    avg_coverage = sum(coverage_by_shift.values()) / len(coverage_by_shift)
                    st.metric("Average Coverage", f"{avg_coverage:.1f}%")

                # Understaffing summary
                understaffing = results.get('understaffing', [])
                if understaffing:
                    st.subheader("❌ Understaffed Shifts")
                    understaffing_df = pd.DataFrame(understaffing)
                    st.dataframe(understaffing_df, use_container_width=True)

            # Executive summary
            with st.expander("📋 Executive Summary"):
                st.write("**Optimization Results:**")
                st.write(f"- Status: {results.get('status', 'Unknown')}")
                obj_val = results.get('objective_value', 0) or 0
                st.write(f"- Objective Value: EGP{obj_val:,.0f}")
                st.write(f"- Total Assignments: {len(results.get('assignments', []))}")
                st.write(f"- Understaffed Shifts: {len(results.get('understaffing', []))}")
                solve_time = results.get('solve_time', 0) or 0
                st.write(f"- Solve Time: {solve_time:.2f} seconds")
        else:
            st.info("📋 Run optimization to see scenario results")

def show_results_tab():
    """Results and reporting tab."""
    st.header("📋 Results & Analytics")

    if not st.session_state.optimization_results:
        st.info("📋 Run optimization to see results")
        return

    results = st.session_state.optimization_results
    solver = st.session_state.solver

    # Results summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Status", results.get('status', 'UNKNOWN'))

    with col2:
        obj_value = results.get('objective_value', 0) or 0
        st.metric("Total Cost", f"EGP{obj_value:,.0f}")

    with col3:
        total_assignments = len(results.get('assignments', []))
        st.metric("Assignments", total_assignments)

    with col4:
        understaffed = len(results.get('understaffing', []))
        st.metric("Understaffed", understaffed)

    # Detailed results tabs
    result_tabs = st.tabs(["📊 Overview", "👥 Assignments", "⚠️ Issues", "💰 Cost Analysis", "📁 Downloads"])

    with result_tabs[0]:  # Overview
        st.subheader("📊 Results Overview")

        # Show aggregated results report at the top
        if results.get('assignments'):
            report_gen = ReportGenerator(solver, results)
            aggregated_df = report_gen.generate_aggregated_results()

            if not aggregated_df.empty:
                st.subheader("Aggregated Results Report")


                # Hide the state_raw column from display
                display_df = aggregated_df.drop(columns=['state_raw']) if 'state_raw' in aggregated_df.columns else aggregated_df
                st.dataframe(display_df, use_container_width=True)



                # Add summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_shifts = len(aggregated_df)
                    st.metric("Total Shifts", total_shifts)
                with col2:
                    fulfilled_count = len(aggregated_df[aggregated_df['state_raw'] == 'Fulfilled'])
                    st.metric("Fulfilled Shifts", fulfilled_count)
                with col3:
                    understaffed_count = len(aggregated_df[aggregated_df['state_raw'] == 'Understaffed'])
                    st.metric("Understaffed Shifts", understaffed_count, delta=f"-{understaffed_count}" if understaffed_count > 0 else None)
                with col4:
                    overstaffed_count = len(aggregated_df[aggregated_df['state_raw'] == 'Overstaffed'])
                    st.metric("Overstaffed Shifts", overstaffed_count, delta=f"+{overstaffed_count}" if overstaffed_count > 0 else None)


                st.markdown("---")  # Add separator

        if results.get('assignments'):
            assignments_df = pd.DataFrame(results['assignments'])

            # Assignment by date chart
            assignments_df['date'] = pd.to_datetime(assignments_df['date'])
            daily_assignments = assignments_df.groupby(assignments_df['date'].dt.date).size()

            fig = px.bar(
                x=daily_assignments.index,
                y=daily_assignments.values,
                title="Daily Assignment Count"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Assignment by shift
            shift_assignments = assignments_df['shift_id'].value_counts()
            fig2 = px.pie(
                values=shift_assignments.values,
                names=shift_assignments.index,
                title="Assignments by Shift Type"
            )
            st.plotly_chart(fig2, use_container_width=True)

    with result_tabs[1]:  # Assignments
            st.subheader("👥 All Assignments")

            if results.get('assignments'):
                assignments_df = pd.DataFrame(results['assignments'])
                assignments_df['date'] = pd.to_datetime(assignments_df['date'])

                # Add employee names
                if solver and not solver.employees.empty:
                    emp_names = solver.employees.set_index('employee_id')['name'].to_dict()
                    assignments_df['employee_name'] = assignments_df['employee_id'].map(emp_names)

                # Display assignments table
                display_df = assignments_df.copy()
                display_df['date'] = display_df['date'].dt.date
                st.dataframe(display_df, use_container_width=True)

                # Add Gantt Chart section
                st.subheader("📊 Schedule Gantt Chart")

                # Filters for Gantt chart
                col1, col2 = st.columns(2)

                with col1:
                    employees = ["All"] + sorted(assignments_df['employee_id'].unique().tolist())
                    selected_employee = st.selectbox("Filter by Employee:", employees, key="gantt_employee")

                with col2:
                    locations = ["All"] + sorted(assignments_df['location_id'].unique().tolist())
                    selected_location = st.selectbox("Filter by Location:", locations, key="gantt_location")

                # Create and display Gantt chart
                try:
                    if st.session_state.optimization_results and 'assignments' in st.session_state.optimization_results:
                        # Create report generator instance for Gantt chart
                        report_gen = ReportGenerator(solver, results)

                        gantt_fig = report_gen.create_gantt_chart(
                            assignments_df,
                            solver,
                            filter_employee=selected_employee,
                            filter_location=selected_location
                        )

                        if gantt_fig:
                            st.plotly_chart(gantt_fig, use_container_width=True)
                        else:
                            st.info("No data to display for selected filters")
                    else:
                        st.info("No assignment data available for Gantt chart")

                except Exception as e:
                    st.error(f"Error creating Gantt chart: {str(e)}")
                    st.info("Displaying assignments table only")

            # Assignment statistics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Assignment Stats")
                st.write(f"Total assignments: {len(assignments_df)}")
                st.write(f"Unique employees: {assignments_df['employee_id'].nunique()}")
                st.write(f"Date range: {assignments_df['date'].min()} to {assignments_df['date'].max()}")

            with col2:
                st.subheader("⏰ Hours Summary")
                total_hours = assignments_df['hours'].sum()
                avg_hours_per_assignment = assignments_df['hours'].mean()
                st.write(f"Total hours: {total_hours:,.1f}")
                st.write(f"Average per assignment: {avg_hours_per_assignment:.1f}")

    with result_tabs[2]:  # Issues
        st.subheader("⚠️ Issues & Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Understaffing")
            understaffing = results.get('understaffing', [])
            if understaffing:
                understaffing_df = pd.DataFrame(understaffing)
                understaffing_df['date'] = pd.to_datetime(understaffing_df['date']).dt.date
                st.error(f"❌ {len(understaffing)} shifts are understaffed")
                st.dataframe(understaffing_df, use_container_width=True)
            else:
                st.success("✅ No understaffed shifts!")

        with col2:
            st.subheader("📈 Overstaffing")
            overstaffing = results.get('overstaffing', [])
            if overstaffing:
                overstaffing_df = pd.DataFrame(overstaffing)
                overstaffing_df['date'] = pd.to_datetime(overstaffing_df['date']).dt.date
                st.warning(f"⚠️ {len(overstaffing)} shifts are overstaffed")
                st.dataframe(overstaffing_df, use_container_width=True)
            else:
                st.success("✅ No overstaffed shifts!")

    with result_tabs[3]:  # Cost Analysis
        st.subheader("💰 Cost Breakdown")

        if results.get('assignments') and solver:
            assignments_df = pd.DataFrame(results['assignments'])

            # Calculate costs
            total_labor_cost = 0
            cost_by_employee = {}

            for _, assignment in assignments_df.iterrows():
                emp_data = solver.employees[solver.employees['employee_id'] == assignment['employee_id']].iloc[0]
                cost = assignment['hours'] * emp_data['hourly_rate']
                total_labor_cost += cost

                if assignment['employee_id'] not in cost_by_employee:
                    cost_by_employee[assignment['employee_id']] = 0
                cost_by_employee[assignment['employee_id']] += cost

            # Display costs
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Labor Cost", f"EGP{total_labor_cost:,.2f}")

                # Overtime costs
                overtime_cost = 0
                for emp_id, ot_hours in results.get('overtime', {}).items():
                    emp_data = solver.employees[solver.employees['employee_id'] == emp_id].iloc[0]
                    ot_cost = ot_hours * emp_data['hourly_rate'] * emp_data['overtime_multiplier']
                    overtime_cost += ot_cost

                st.metric("Overtime Cost", f"EGP{overtime_cost:,.2f}")

            with col2:
                # Top cost employees
                top_costs = sorted(cost_by_employee.items(), key=lambda x: x[1], reverse=True)[:10]

                st.subheader("💰 Top Cost Employees")
                for emp_id, cost in top_costs:
                    st.write(f"{emp_id}: EGP{cost:,.2f}")

    with result_tabs[4]:  # Downloads
        st.subheader("📁 Download Results")

        # File download buttons
        output_files = [
            ('aggregated_results.xlsx', 'Aggregated Results Report'),
            ('final_schedule.xlsx', 'Final Schedule'),
            ('payroll_summary.xlsx', 'Payroll Summary'),
            ('changelog.xlsx', 'Change Log'),
            ('insights.json', 'Insights Report')
        ]


        for filename, display_name in output_files:
            filepath = f'outputs/{filename}'
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    st.download_button(
                        label=f"📥 {display_name}",
                        data=f.read(),
                        file_name=filename,
                        mime='application/octet-stream'
                    )
            else:
                st.write(f"❌ {display_name} not available")

        # Charts downloads
        st.subheader("📊 Charts")
        chart_files = [
            ('coverage_by_store.png', 'Coverage by Store'),
            ('cost_breakdown.png', 'Cost Breakdown'),
            ('top_understaffed.png', 'Top Understaffed')
        ]

        for filename, display_name in chart_files:
            filepath = f'outputs/charts/{filename}'
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    st.download_button(
                        label=f"🖼️ {display_name}",
                        data=f.read(),
                        file_name=filename,
                        mime='image/png'
                    )
            else:
                st.write(f"❌ {display_name} not available")

def main():
    """Main application entry point."""
    try:
        create_streamlit_app()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
