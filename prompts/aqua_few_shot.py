"""
prompts/aqua_few_shot.py
─────────────────────────
Few-shot exemplars for AQuA-RAT (algebraic multiple-choice questions).
Uses SymPy's solve_it for symbolic algebra — the paper's key trick for
polynomial equations that CoT simply cannot handle.
"""

AQUA_EXEMPLARS = [
    (
        "In a flight of 600 km, an aircraft was slowed down due to bad weather. "
        "Its average speed for the trip was reduced by 200 km/hr and the time of "
        "flight increased by 30 minutes. The duration of the flight is:",
        ["A)1 hour", "B)2 hours", "C)3 hours", "D)4 hours", "E)5 hours"],
        """\
from sympy import Symbol
from sympy import solve_it
duration = Symbol('duration', positive=True)
delay = 30 / 60
total_distance = 600
original_speed = total_distance / duration
reduced_speed = total_distance / (duration + delay)
solution = solve_it(original_speed - reduced_speed - 200, duration)
ans = solution[duration]"""
    ),
    (
        "M men agree to purchase a gift for Rs. D. If 3 men drop out how much more "
        "will each have to contribute towards the purchase of the gift?",
        ["A)D/(M-3)", "B)MD/3", "C)M/(D-3)", "D)3D/(M2-3M)", "E)None of these"],
        """\
from sympy import Symbol, simplify
M = Symbol('M')
D = Symbol('D')
cost_before_dropout = D / M
cost_after_dropout = D / (M - 3)
ans = simplify(cost_after_dropout - cost_before_dropout)"""
    ),
    (
        "A sum of money at simple interest amounts to Rs. 815 in 3 years and to "
        "Rs. 854 in 4 years. The sum is:",
        ["A)Rs. 650", "B)Rs. 690", "C)Rs. 698", "D)Rs. 700", "E)None of these"],
        """\
from sympy import Symbol
from sympy import solve_it
deposit = Symbol('deposit', positive=True)
interest = Symbol('interest', positive=True)
money_in_3_years = deposit + 3 * interest
money_in_4_years = deposit + 4 * interest
solution = solve_it([money_in_3_years - 815, money_in_4_years - 854], [deposit, interest])
ans = solution[deposit]"""
    ),
    (
        "Find out which of the following values is the multiple of X, "
        "if it is divisible by 9 and 12?",
        ["A)36", "B)15", "C)17", "D)5", "E)7"],
        """\
options = [36, 15, 17, 5, 7]
for option in options:
    if option % 9 == 0 and option % 12 == 0:
        ans = option
        break"""
    ),
    (
        "35% of the employees of a company are men. 60% of the men in the company "
        "speak French and 40% of the employees of the company speak French. "
        "What is % of the women in the company who do not speak French?",
        ["A)4%", "B)10%", "C)96%", "D)90.12%", "E)70.77%"],
        """\
num_women = 65
men_speaking_french = 0.6 * 35
employees_speaking_french = 0.4 * 100
women_speaking_french = employees_speaking_french - men_speaking_french
women_not_speaking_french = num_women - women_speaking_french
ans = women_not_speaking_french / num_women * 100"""
    ),
    (
        "In one hour, a boat goes 11 km/hr along the stream and 5 km/hr against "
        "the stream. The speed of the boat in still water (in km/hr) is:",
        ["A)4 kmph", "B)5 kmph", "C)6 kmph", "D)7 kmph", "E)8 kmph"],
        """\
from sympy import Symbol
from sympy import solve_it
boat_speed = Symbol('boat_speed', positive=True)
stream_speed = Symbol('stream_speed', positive=True)
along_stream_speed = 11
against_stream_speed = 5
solution = solve_it(
    [boat_speed + stream_speed - along_stream_speed,
     boat_speed - stream_speed - against_stream_speed],
    [boat_speed, stream_speed]
)
ans = solution[boat_speed]"""
    ),
    (
        "The difference between simple interest and compound interest at the same "
        "rate for Rs.5000 for 2 years is Rs.72. The rate of interest is?",
        ["A)10%", "B)12%", "C)6%", "D)8%", "E)4%"],
        """\
from sympy import Symbol
from sympy import solve_it
interest_rate = Symbol('interest_rate', positive=True)
amount = 5000
amount_with_simple_interest = amount * (1 + 2 * interest_rate / 100)
amount_with_compound_interest = amount * (1 + interest_rate / 100) ** 2
solution = solve_it(amount_with_compound_interest - amount_with_simple_interest - 72, interest_rate)
ans = solution[interest_rate]"""
    ),
    (
        "The area of a rectangle is 15 square centimeters and the perimeter is "
        "16 centimeters. What are the dimensions of the rectangle?",
        ["A)2&4", "B)3&5", "C)4&6", "D)5&7", "E)6&8"],
        """\
from sympy import Symbol
from sympy import solve_it
width = Symbol('width', positive=True)
height = Symbol('height', positive=True)
area = 15
perimeter = 16
solution = solve_it([width * height - area, 2 * (width + height) - perimeter], [width, height])
ans = (solution[width], solution[height])"""
    ),
]


def build_aqua_prompt(question: str, options: list[str], n_shots: int = 8) -> str:
    """
    Build few-shot prompt for AQuA. Includes answer options in each exemplar.
    """
    exemplars = AQUA_EXEMPLARS[:n_shots]
    lines = [
        "# Write Python Code to solve the following questions.",
        "# Store your result as a variable named 'ans'.",
        "# Return only the Python code for the final question. Do not repeat examples.",
        "from sympy import Symbol, symbols, simplify, solve",
        "",
    ]

    for q, opts, prog in exemplars:
        opts_str = str(opts)
        lines.append(f"# Question: {q}")
        lines.append(f"# Answer options: {opts_str}")
        lines.append(prog)
        lines.append("")

    opts_str = str(options)
    lines.append(f"# Question: {question}")
    lines.append(f"# Answer options: {opts_str}")
    lines.append("# Python code for this question only:")
    return "\n".join(lines)
