"""
prompts/gsm8k_few_shot.py
──────────────────────────
Few-shot exemplars for GSM8K, taken directly from the paper's appendix.
These use semantically meaningful variable names (a core PoT design choice).
"""

# Each exemplar is a (question, program) pair.
GSM8K_EXEMPLARS = [
    (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
        "morning and bakes muffins for her friends every day with four. She sells "
        "the remainder at the farmers' market daily for $2 per fresh duck egg. "
        "How much in dollars does she make every day at the farmers' market?",
        """\
# Python code, return ans
total_eggs = 16
eaten_eggs = 3
baked_eggs = 4
sold_eggs = total_eggs - eaten_eggs - baked_eggs
dollars_per_egg = 2
ans = sold_eggs * dollars_per_egg"""
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?",
        """\
# Python code, return ans
bolts_of_blue_fiber = 2
bolts_of_white_fiber = bolts_of_blue_fiber / 2
ans = bolts_of_blue_fiber + bolts_of_white_fiber"""
    ),
    (
        "Josh decides to try flipping a house. He buys a house for $80,000 and "
        "then puts in $50,000 in repairs. This increased the value of the house "
        "by 150%. How much profit did he make?",
        """\
# Python code, return ans
cost_of_original_house = 80000
increase_rate = 150 / 100
value_of_house = (1 + increase_rate) * cost_of_original_house
cost_of_repair = 50000
ans = value_of_house - cost_of_repair - cost_of_original_house"""
    ),
    (
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken "
        "feed. She gives the chickens their feed in three separate meals. In the "
        "morning she gives her flock 15 cups of feed. In the afternoon she gives "
        "her chickens another 25 cups of feed. How many cups of feed does she need "
        "to give her chickens in the final meal of the day if the size of Wendi's "
        "flock is 20 chickens?",
        """\
# Python code, return ans
num_of_chickens = 20
cups_for_each_chicken = 3
cups_for_all_chicken = num_of_chickens * cups_for_each_chicken
cups_in_the_morning = 15
cups_in_the_afternoon = 25
ans = cups_for_all_chicken - cups_in_the_morning - cups_in_the_afternoon"""
    ),
    (
        "Kylar went to the store to buy glasses for his new apartment. One glass "
        "costs $5, but every second glass costs only 60% of the price. Kylar wants "
        "to buy 16 glasses. How much does he need to pay for them?",
        """\
# Python code, return ans
num_glasses = 16
first_glass_cost = 5
second_glass_cost = 5 * 0.6
ans = 0
for i in range(num_glasses):
    if i % 2 == 0:
        ans += first_glass_cost
    else:
        ans += second_glass_cost"""
    ),
    (
        "Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 "
        "miles, then another hour to walk the next 2 miles. If she wants her "
        "average speed to be 4 miles per hour, what speed (in miles per hour) does "
        "she need to walk the remaining distance?",
        """\
# Python code, return ans
average_mile_per_hour = 4
total_trail_miles = 12
remaining_miles = total_trail_miles - 4 - 2
total_hours = total_trail_miles / average_mile_per_hour
remaining_hours = total_hours - 2
ans = remaining_miles / remaining_hours"""
    ),
    (
        "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each "
        "year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 "
        "a year to water and feed the tree. How many years will it take before he "
        "starts earning money on the lemon tree?",
        """\
# Python code, return ans
total_cost = 90
cost_of_watering_and_feeding = 3
cost_of_each_lemon = 1.5
num_of_lemon_per_year = 7
ans = 0
while total_cost > 0:
    total_cost += cost_of_watering_and_feeding
    total_cost -= num_of_lemon_per_year * cost_of_each_lemon
    ans += 1"""
    ),
    (
        "Jordan wanted to surprise her mom with a homemade birthday cake. From "
        "reading the instructions she knew it would take 20 minutes to make the "
        "cake batter and 30 minutes to bake the cake. The cake would require 2 "
        "hours to cool and an additional 10 minutes to frost the cake. If she "
        "plans to make the cake all on the same day, what is the latest time of "
        "day that Jordan can start making the cake to be ready to serve it at 5:00 pm?",
        """\
# Python code, return ans
minutes_to_make_batter = 20
minutes_to_bake_cake = 30
minutes_to_cool_cake = 2 * 60
minutes_to_frost_cake = 10
total_minutes = minutes_to_make_batter + minutes_to_bake_cake + minutes_to_cool_cake + minutes_to_frost_cake
total_hours = total_minutes / 60
ans = 5 - total_hours"""
    ),
]


def build_few_shot_prompt(question: str, n_shots: int = 8) -> str:
    """
    Construct the few-shot prompt for GSM8K.

    Parameters
    ----------
    question : The test question to answer.
    n_shots  : Number of exemplars to include (1-8).
    """
    exemplars = GSM8K_EXEMPLARS[:n_shots]
    prompt_parts = []

    for q, prog in exemplars:
        prompt_parts.append(f"Question: {q}\n{prog}")

    prompt_parts.append(f"Question: {question}\n# Python code, return ans")
    return "\n\n".join(prompt_parts)