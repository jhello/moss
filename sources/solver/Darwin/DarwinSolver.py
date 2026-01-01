from data import Case
from solver import Status


class DarwinSolver:
    def __init__(self, case: Case):
        self.case = case
        self.status = Status.NOT_STARTED

    def solve(self):
        print("Solving using DarwinSolver")
        #TODO: Implement the actual solving logic here


        self.status = Status.SUCCESS

    def get_status(self):
        return self.status