"""
Analysis module for convergence and error analysis.

Contains test cases with exact solutions for verification:

2D Test Cases (D'Yakonov ADI Solver):
- test_case_1_decaying_bubble_2d: Dirichlet BCs, zero forcing
- test_case_2_standing_wave_2d: Neumann BCs, inhomogeneous forcing
- test_case_3_quadratic_decay_2d: Robin BCs, inhomogeneous forcing

3D Test Cases (Douglas-Gunn ADI Solver):
- test_case_1_decaying_bubble: Dirichlet BCs, zero forcing
- test_case_2_standing_wave: Neumann BCs, inhomogeneous forcing
- test_case_3_quadratic_decay: Robin BCs, inhomogeneous forcing

Each test case includes:
- Error comparison against exact analytical solutions
- Spatial convergence analysis (grid refinement study)
- Animated visualization (gif output)
"""
