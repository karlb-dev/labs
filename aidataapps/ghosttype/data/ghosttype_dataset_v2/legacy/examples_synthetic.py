"""Synthetic eval items authored as grok-4.6 reference completions.

Prompts and schema context are simulated in the same MSSQL Copilot ghost-text
format as the traces. Gold SQL uses only objects/columns listed in that context.
"""

SYN = []


def add(**kwargs):
    kwargs.setdefault("kind", "synthetic")
    kwargs.setdefault("gold_author", "grok-4.6")
    kwargs.setdefault("gold_model", "grok-4.6")
    kwargs.setdefault("gold_role", "synthetic")
    kwargs.setdefault("origin_model_id", "grok-4.6")
    kwargs.setdefault("origin_model_family", "grok-4.6")
    kwargs.setdefault("origin_model_vendor", "xai")
    kwargs.setdefault("edit_notes", "Synthetic prompt and grok-4.6 reference completion.")
    SYN.append(kwargs)


# ===========================================================================
# FitnessApp_Test — user-schema intent
# ===========================================================================

add(
    id="syn-intent-fa-001",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- list all users",
    gold="""SELECT
    user_id,
    full_name,
    email,
    date_of_birth,
    gender,
    height_cm,
    weight_kg,
    created_at
FROM dbo.Users
ORDER BY
    full_name;""",
    eval={
        "difficulty": "easy",
        "tags": ["user-schema", "select", "fitnessapp", "privacy"],
        "must_contain": ["dbo.Users", "full_name", "email"],
        "must_not_contain": ["password_hash"],
        "required_objects": ["dbo.Users"],
        "rubric": "List users from dbo.Users. Do not project password_hash unless asked.",
    },
)

add(
    id="syn-intent-fa-002",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- meals over 600 calories",
    gold="""SELECT
    meal_id,
    meal_name,
    category,
    calories,
    protein_g,
    carbs_g,
    fats_g,
    dietary_preference
FROM dbo.Meals
WHERE calories > 600
ORDER BY
    calories DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["filter", "meals"],
        "must_contain": ["dbo.Meals", "calories"],
        "required_objects": ["dbo.Meals"],
        "rubric": "Filter dbo.Meals by calories > 600 using the listed calories column.",
    },
)

add(
    id="syn-intent-fa-003",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- vegan or vegetarian meals",
    gold="""SELECT
    meal_id,
    meal_name,
    category,
    calories,
    dietary_preference
FROM dbo.Meals
WHERE dietary_preference IN (N'vegan', N'vegetarian')
ORDER BY
    meal_name;""",
    eval={
        "difficulty": "easy",
        "tags": ["filter", "meals", "predicate"],
        "must_contain": ["dbo.Meals", "dietary_preference"],
        "required_objects": ["dbo.Meals"],
        "rubric": "Use dbo.Meals.dietary_preference. Do not invent a boolean is_vegan column.",
    },
)

add(
    id="syn-intent-fa-004",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- total completed payments per user",
    gold="""SELECT
    u.user_id,
    u.full_name,
    u.email,
    SUM(p.amount) AS total_paid
FROM dbo.Users AS u
INNER JOIN dbo.Payments AS p
    ON p.user_id = u.user_id
WHERE p.payment_status = N'completed'
GROUP BY
    u.user_id,
    u.full_name,
    u.email
ORDER BY
    total_paid DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["join", "aggregate", "payments"],
        "must_contain": ["dbo.Payments", "dbo.Users", "SUM"],
        "required_objects": ["dbo.Payments", "dbo.Users"],
        "rubric": "Join Users to Payments on user_id. Aggregate amount. Prefer filtering payment_status rather than inventing extra columns.",
    },
)

add(
    id="syn-intent-fa-005",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- challenge leaderboard with user names",
    gold="""SELECT
    c.challenge_id,
    c.challenge_name,
    u.user_id,
    u.full_name,
    cp.progress,
    cp.rank
FROM dbo.Challenges AS c
INNER JOIN dbo.Challenge_Participants AS cp
    ON cp.challenge_id = c.challenge_id
INNER JOIN dbo.Users AS u
    ON u.user_id = cp.user_id
ORDER BY
    c.challenge_name,
    cp.rank;""",
    eval={
        "difficulty": "medium",
        "tags": ["join", "leaderboard", "fk"],
        "must_contain": ["dbo.Challenge_Participants", "dbo.Challenges", "dbo.Users"],
        "required_objects": ["dbo.Challenge_Participants", "dbo.Challenges", "dbo.Users"],
        "rubric": "Use the listed FK path Challenges -> Challenge_Participants -> Users. Rank/progress columns exist; do not invent points.",
    },
)

add(
    id="syn-intent-fa-006",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- workouts that require dumbbells",
    gold="""SELECT
    workout_id,
    workout_name,
    category,
    difficulty_level,
    equipment_required,
    duration_minutes
FROM dbo.Workouts
WHERE equipment_required LIKE N'%dumbbell%'
ORDER BY
    workout_name;""",
    eval={
        "difficulty": "easy",
        "tags": ["filter", "like", "workouts"],
        "must_contain": ["dbo.Workouts", "equipment_required"],
        "required_objects": ["dbo.Workouts"],
        "rubric": "Filter dbo.Workouts.equipment_required. Do not invent an equipment child table.",
    },
)

add(
    id="syn-intent-fa-007",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- wearable step counts in the last 7 days",
    gold="""SELECT
    u.user_id,
    u.full_name,
    w.device_type,
    w.steps,
    w.distance_km,
    w.heart_rate,
    w.recorded_at
FROM dbo.Wearable_Data AS w
INNER JOIN dbo.Users AS u
    ON u.user_id = w.user_id
WHERE w.recorded_at >= DATEADD(DAY, -7, GETDATE())
ORDER BY
    w.recorded_at DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["date-filter", "wearable", "join"],
        "must_contain": ["dbo.Wearable_Data", "DATEADD", "steps"],
        "required_objects": ["dbo.Wearable_Data"],
        "rubric": "Use Wearable_Data.recorded_at and steps. Join Users for names via listed FK.",
    },
)

add(
    id="syn-intent-fa-008",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- users who have never logged a workout",
    gold="""SELECT
    u.user_id,
    u.full_name,
    u.email,
    u.created_at
FROM dbo.Users AS u
WHERE NOT EXISTS (
    SELECT 1
    FROM dbo.User_Workouts AS uw
    WHERE uw.user_id = u.user_id
)
ORDER BY
    u.created_at;""",
    eval={
        "difficulty": "medium",
        "tags": ["anti-join", "not-exists"],
        "must_contain": ["dbo.Users", "dbo.User_Workouts"],
        "must_contain_any": ["NOT EXISTS", "LEFT JOIN"],
        "required_objects": ["dbo.Users", "dbo.User_Workouts"],
        "rubric": "Users with no User_Workouts rows. NOT EXISTS or LEFT JOIN ... IS NULL. Do not invent a last_workout column.",
    },
)

add(
    id="syn-intent-fa-009",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- average calories burned per user across workout sessions",
    gold="""SELECT
    u.user_id,
    u.full_name,
    AVG(ws.calories_burned) AS avg_calories_burned,
    COUNT(*) AS session_count
FROM dbo.Users AS u
INNER JOIN dbo.User_Workouts AS uw
    ON uw.user_id = u.user_id
INNER JOIN dbo.Workout_Sessions AS ws
    ON ws.user_workout_id = uw.user_workout_id
GROUP BY
    u.user_id,
    u.full_name
ORDER BY
    avg_calories_burned DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["join", "aggregate", "sessions"],
        "must_contain": ["dbo.Workout_Sessions", "calories_burned", "AVG"],
        "required_objects": ["dbo.Workout_Sessions", "dbo.User_Workouts", "dbo.Users"],
        "rubric": "Path Users -> User_Workouts -> Workout_Sessions. Aggregate calories_burned.",
    },
)

add(
    id="syn-intent-fa-010",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- active fitness goals with target weight",
    gold="""SELECT
    g.goal_id,
    u.full_name,
    g.goal_type,
    g.target_weight,
    g.weekly_target,
    g.created_at,
    g.status
FROM dbo.User_Fitness_Goals AS g
INNER JOIN dbo.Users AS u
    ON u.user_id = g.user_id
WHERE g.status = N'active'
ORDER BY
    g.created_at DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["goals", "filter"],
        "must_contain": ["dbo.User_Fitness_Goals", "status"],
        "required_objects": ["dbo.User_Fitness_Goals"],
        "rubric": "Filter User_Fitness_Goals.status. target_weight is listed.",
    },
)

add(
    id="syn-intent-fa-011",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- count users by gender",
    gold="""SELECT
    gender,
    COUNT(*) AS user_count
FROM dbo.Users
GROUP BY
    gender
ORDER BY
    user_count DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["aggregate", "group-by"],
        "must_contain": ["dbo.Users", "gender", "COUNT"],
        "required_objects": ["dbo.Users"],
        "rubric": "GROUP BY dbo.Users.gender.",
    },
)

add(
    id="syn-intent-fa-012",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- high protein low carb meals",
    gold="""SELECT
    meal_id,
    meal_name,
    protein_g,
    carbs_g,
    fats_g,
    calories,
    dietary_preference
FROM dbo.Meals
WHERE protein_g > carbs_g
ORDER BY
    protein_g DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["predicate", "meals"],
        "must_contain": ["dbo.Meals", "protein_g", "carbs_g"],
        "required_objects": ["dbo.Meals"],
        "rubric": "Compare listed protein_g and carbs_g. Do not invent net_carbs or fiber.",
    },
)

add(
    id="syn-intent-fa-013",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- latest AI recommendation per user",
    gold="""SELECT
    r.recommendation_id,
    r.user_id,
    u.full_name,
    r.type,
    r.generated_text,
    r.created_at
FROM dbo.AI_Recommendations AS r
INNER JOIN dbo.Users AS u
    ON u.user_id = r.user_id
WHERE r.created_at = (
    SELECT MAX(r2.created_at)
    FROM dbo.AI_Recommendations AS r2
    WHERE r2.user_id = r.user_id
)
ORDER BY
    r.created_at DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["window-or-subquery", "latest-per-group"],
        "must_contain": ["dbo.AI_Recommendations", "created_at"],
        "required_objects": ["dbo.AI_Recommendations"],
        "rubric": "One latest recommendation per user using created_at. ROW_NUMBER or correlated MAX are both fine. Do not invent a is_latest flag.",
    },
)

add(
    id="syn-intent-fa-014",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- wearable readings with heart rate over 160",
    gold="""SELECT
    w.wearable_id,
    u.full_name,
    w.device_type,
    w.heart_rate,
    w.steps,
    w.recorded_at
FROM dbo.Wearable_Data AS w
INNER JOIN dbo.Users AS u
    ON u.user_id = w.user_id
WHERE w.heart_rate > 160
ORDER BY
    w.heart_rate DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["filter", "wearable"],
        "must_contain": ["dbo.Wearable_Data", "heart_rate"],
        "required_objects": ["dbo.Wearable_Data"],
        "rubric": "Filter Wearable_Data.heart_rate. Column is listed as int.",
    },
)

add(
    id="syn-intent-fa-015",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- hardest and longest workouts",
    gold="""SELECT
    workout_id,
    workout_name,
    category,
    difficulty_level,
    duration_minutes,
    equipment_required
FROM dbo.Workouts
ORDER BY
    CASE difficulty_level
        WHEN N'expert' THEN 4
        WHEN N'hard' THEN 3
        WHEN N'intermediate' THEN 2
        WHEN N'easy' THEN 1
        ELSE 0
    END DESC,
    duration_minutes DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["order-by", "workouts"],
        "must_contain": ["dbo.Workouts", "difficulty_level", "duration_minutes"],
        "required_objects": ["dbo.Workouts"],
        "rubric": "Order dbo.Workouts by difficulty_level and duration_minutes. Do not invent a difficulty_rank column on the table.",
    },
)

add(
    id="syn-intent-fa-016",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- duplicate email addresses in users",
    gold="""SELECT
    email,
    COUNT(*) AS user_count
FROM dbo.Users
GROUP BY
    email
HAVING COUNT(*) > 1
ORDER BY
    user_count DESC,
    email;""",
    eval={
        "difficulty": "easy",
        "tags": ["duplicate", "having"],
        "must_contain": ["dbo.Users", "email", "HAVING"],
        "required_objects": ["dbo.Users"],
        "rubric": "GROUP BY email HAVING COUNT(*) > 1.",
    },
)

add(
    id="syn-intent-fa-017",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- users who have paid but have no fitness goals",
    gold="""SELECT
    u.user_id,
    u.full_name,
    u.email
FROM dbo.Users AS u
WHERE EXISTS (
    SELECT 1
    FROM dbo.Payments AS p
    WHERE p.user_id = u.user_id
)
AND NOT EXISTS (
    SELECT 1
    FROM dbo.User_Fitness_Goals AS g
    WHERE g.user_id = u.user_id
)
ORDER BY
    u.full_name;""",
    eval={
        "difficulty": "medium",
        "tags": ["exists", "anti-join"],
        "must_contain": ["dbo.Payments", "dbo.User_Fitness_Goals", "dbo.Users"],
        "required_objects": ["dbo.Users", "dbo.Payments", "dbo.User_Fitness_Goals"],
        "rubric": "Users with at least one payment and zero goals. Use listed FKs only.",
    },
)

add(
    id="syn-intent-fa-018",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- total planned workout minutes per user",
    gold="""SELECT
    u.user_id,
    u.full_name,
    SUM(w.duration_minutes) AS total_minutes
FROM dbo.Users AS u
INNER JOIN dbo.User_Workouts AS uw
    ON uw.user_id = u.user_id
INNER JOIN dbo.Workouts AS w
    ON w.workout_id = uw.workout_id
GROUP BY
    u.user_id,
    u.full_name
ORDER BY
    total_minutes DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["join", "aggregate"],
        "must_contain": ["dbo.Workouts", "duration_minutes", "dbo.User_Workouts"],
        "required_objects": ["dbo.Users", "dbo.User_Workouts", "dbo.Workouts"],
        "rubric": "Sum Workouts.duration_minutes via User_Workouts. Do not invent a minutes column on User_Workouts.",
    },
)

add(
    id="syn-intent-fa-019",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- open challenges that have not ended yet",
    gold="""SELECT
    challenge_id,
    challenge_name,
    start_date,
    end_date
FROM dbo.Challenges
WHERE end_date >= CAST(GETDATE() AS date)
ORDER BY
    start_date;""",
    eval={
        "difficulty": "easy",
        "tags": ["date-filter", "challenges"],
        "must_contain": ["dbo.Challenges", "end_date"],
        "required_objects": ["dbo.Challenges"],
        "rubric": "Filter Challenges.end_date. Do not invent an is_open column.",
    },
)

add(
    id="syn-intent-fa-020",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- average calories by meal category",
    gold="""SELECT
    category,
    COUNT(*) AS meal_count,
    AVG(calories) AS avg_calories,
    AVG(protein_g) AS avg_protein_g
FROM dbo.Meals
GROUP BY
    category
ORDER BY
    avg_calories DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["aggregate", "meals"],
        "must_contain": ["dbo.Meals", "category", "AVG"],
        "required_objects": ["dbo.Meals"],
        "rubric": "GROUP BY Meals.category with AVG(calories).",
    },
)

add(
    id="syn-intent-fa-021",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- users created in the last 30 days",
    gold="""SELECT
    user_id,
    full_name,
    email,
    created_at
FROM dbo.Users
WHERE created_at >= DATEADD(DAY, -30, GETDATE())
ORDER BY
    created_at DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["date-filter", "users"],
        "must_contain": ["dbo.Users", "DATEADD"],
        "required_objects": ["dbo.Users"],
        "rubric": "Filter Users.created_at with DATEADD.",
    },
)

add(
    id="syn-intent-fa-022",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- session feedback that is not empty",
    gold="""SELECT
    ws.session_id,
    ws.session_date,
    ws.calories_burned,
    ws.reps_completed,
    ws.feedback,
    u.full_name
FROM dbo.Workout_Sessions AS ws
INNER JOIN dbo.User_Workouts AS uw
    ON uw.user_workout_id = ws.user_workout_id
INNER JOIN dbo.Users AS u
    ON u.user_id = uw.user_id
WHERE ws.feedback IS NOT NULL
  AND LTRIM(RTRIM(ws.feedback)) <> N''
ORDER BY
    ws.session_date DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["null-filter", "sessions"],
        "must_contain": ["dbo.Workout_Sessions", "feedback"],
        "required_objects": ["dbo.Workout_Sessions"],
        "rubric": "Use Workout_Sessions.feedback. Join back to Users through User_Workouts.",
    },
)

add(
    id="syn-intent-fa-023",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- top paying users by completed payment amount",
    gold="""SELECT TOP (10)
    u.user_id,
    u.full_name,
    SUM(p.amount) AS total_paid,
    COUNT(*) AS payment_count
FROM dbo.Users AS u
INNER JOIN dbo.Payments AS p
    ON p.user_id = u.user_id
WHERE p.payment_status = N'completed'
GROUP BY
    u.user_id,
    u.full_name
ORDER BY
    total_paid DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["top-n", "payments"],
        "must_contain": ["dbo.Payments", "TOP", "SUM"],
        "required_objects": ["dbo.Payments", "dbo.Users"],
        "rubric": "TOP paying users from Payments.amount. Do not invent a lifetime_value column.",
    },
)

add(
    id="syn-intent-fa-024",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- body mass index from user height and weight",
    gold="""SELECT
    user_id,
    full_name,
    height_cm,
    weight_kg,
    CASE
        WHEN height_cm IS NULL OR height_cm = 0 THEN NULL
        ELSE weight_kg / POWER(height_cm / 100.0, 2)
    END AS bmi
FROM dbo.Users
ORDER BY
    bmi DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["computed", "bmi"],
        "must_contain": ["dbo.Users", "height_cm", "weight_kg"],
        "must_not_contain": ["bmi"],
        "required_objects": ["dbo.Users"],
        "rubric": "Compute BMI from listed height_cm and weight_kg. There is no bmi column — do not select u.bmi from the table. (Alias BMI in the projection is fine; must_not_contain is checked on a case-sensitive basis in some scorers — prefer computing rather than reading a bmi column.)",
    },
)

# Fix eval for BMI: must_not_contain bmi would fail the alias. Remove that.
SYN[-1]["eval"]["must_not_contain"] = ["```", "dbo.Orders"]
SYN[-1]["eval"]["rubric"] = (
    "Compute BMI from height_cm and weight_kg. Do not invent a stored bmi column read from the table."
)

add(
    id="syn-intent-fa-025",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- AI recommendations of type workout",
    gold="""SELECT
    r.recommendation_id,
    u.full_name,
    r.type,
    r.generated_text,
    r.created_at
FROM dbo.AI_Recommendations AS r
INNER JOIN dbo.Users AS u
    ON u.user_id = r.user_id
WHERE r.type = N'workout'
ORDER BY
    r.created_at DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["filter", "recommendations"],
        "must_contain": ["dbo.AI_Recommendations", "type"],
        "required_objects": ["dbo.AI_Recommendations"],
        "rubric": "Filter AI_Recommendations.type. Do not invent recommendation subtypes as tables.",
    },
)

add(
    id="syn-intent-fa-026",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- failed payments in the last 14 days",
    gold="""SELECT
    p.payment_id,
    u.full_name,
    p.amount,
    p.payment_method,
    p.payment_status,
    p.payment_date
FROM dbo.Payments AS p
INNER JOIN dbo.Users AS u
    ON u.user_id = p.user_id
WHERE p.payment_status IN (N'failed', N'declined')
  AND p.payment_date >= DATEADD(DAY, -14, GETDATE())
ORDER BY
    p.payment_date DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["payments", "date-filter"],
        "must_contain": ["dbo.Payments", "payment_status"],
        "required_objects": ["dbo.Payments"],
        "rubric": "Filter Payments by status and payment_date. Do not invent an error_message column.",
    },
)

add(
    id="syn-intent-fa-027",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- in-progress user workouts with the workout name",
    gold="""SELECT
    uw.user_workout_id,
    u.full_name,
    w.workout_name,
    w.category,
    uw.start_date,
    uw.end_date,
    uw.status
FROM dbo.User_Workouts AS uw
INNER JOIN dbo.Users AS u
    ON u.user_id = uw.user_id
INNER JOIN dbo.Workouts AS w
    ON w.workout_id = uw.workout_id
WHERE uw.status = N'in_progress'
ORDER BY
    uw.start_date DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["join", "status"],
        "must_contain": ["dbo.User_Workouts", "dbo.Workouts", "status"],
        "required_objects": ["dbo.User_Workouts", "dbo.Workouts", "dbo.Users"],
        "rubric": "Join User_Workouts to Workouts and Users. Filter status with the listed column.",
    },
)

add(
    id="syn-intent-fa-028",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- total distance and steps by device type",
    gold="""SELECT
    device_type,
    COUNT(*) AS reading_count,
    SUM(steps) AS total_steps,
    SUM(distance_km) AS total_distance_km
FROM dbo.Wearable_Data
GROUP BY
    device_type
ORDER BY
    total_steps DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["aggregate", "wearable"],
        "must_contain": ["dbo.Wearable_Data", "device_type", "SUM"],
        "required_objects": ["dbo.Wearable_Data"],
        "rubric": "GROUP BY Wearable_Data.device_type using listed numeric columns.",
    },
)

# ===========================================================================
# ninjadb (Northwind + pubs mix)
# ===========================================================================

add(
    id="syn-intent-nj-001",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- customers in Germany",
    gold="""SELECT
    CustomerID,
    CompanyName,
    ContactName,
    City,
    Region,
    PostalCode,
    Country,
    Phone
FROM dbo.Customers
WHERE Country = N'Germany'
ORDER BY
    City,
    CompanyName;""",
    eval={
        "difficulty": "easy",
        "tags": ["northwind", "filter"],
        "must_contain": ["dbo.Customers", "Germany"],
        "required_objects": ["dbo.Customers"],
        "rubric": "Filter listed Customers.Country.",
    },
)

add(
    id="syn-intent-nj-002",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- discontinued products with category name",
    gold="""SELECT
    p.ProductID,
    p.ProductName,
    c.CategoryName,
    p.UnitPrice,
    p.UnitsInStock,
    p.Discontinued
FROM dbo.Products AS p
INNER JOIN dbo.Categories AS c
    ON c.CategoryID = p.CategoryID
WHERE p.Discontinued = 1
ORDER BY
    p.ProductName;""",
    eval={
        "difficulty": "medium",
        "tags": ["northwind", "join", "products"],
        "must_contain": ["dbo.Products", "Discontinued", "dbo.Categories"],
        "required_objects": ["dbo.Products", "dbo.Categories"],
        "rubric": "Products is detailed on this schema snapshot. Join Categories on CategoryID. Do not use names-only Order Details columns.",
    },
)

add(
    id="syn-intent-nj-003",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- orders that have not shipped yet",
    gold="""SELECT
    o.OrderID,
    o.CustomerID,
    c.CompanyName,
    o.OrderDate,
    o.RequiredDate,
    o.ShippedDate,
    o.Freight,
    o.ShipCity,
    o.ShipRegion
FROM dbo.Orders AS o
INNER JOIN dbo.Customers AS c
    ON c.CustomerID = o.CustomerID
WHERE o.ShippedDate IS NULL
ORDER BY
    o.RequiredDate;""",
    eval={
        "difficulty": "medium",
        "tags": ["northwind", "null-filter", "orders"],
        "must_contain": ["dbo.Orders", "ShippedDate"],
        "required_objects": ["dbo.Orders", "dbo.Customers"],
        "rubric": "Unshipped means ShippedDate IS NULL. Orders.ShipCountry is not listed on this snapshot — do not select it from dbo.Orders.",
    },
)

add(
    id="syn-intent-nj-004",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- authors in California",
    gold="""SELECT
    au_id,
    au_lname,
    au_fname,
    phone,
    address,
    city,
    state,
    zip,
    contract
FROM dbo.authors
WHERE state = 'CA'
ORDER BY
    au_lname,
    au_fname;""",
    eval={
        "difficulty": "easy",
        "tags": ["pubs", "authors"],
        "must_contain": ["dbo.authors", "state"],
        "required_objects": ["dbo.authors"],
        "rubric": "Filter dbo.authors.state. Table is lowercase authors as listed.",
    },
)

add(
    id="syn-intent-nj-005",
    completion_category="intent",
    schema_id="ninjadb_b",
    inferred_system_query=False,
    user_comment="-- sales quantity by store",
    gold="""SELECT
    st.stor_id,
    st.stor_name,
    st.city,
    st.state,
    SUM(s.qty) AS qty_sold
FROM dbo.stores AS st
INNER JOIN dbo.sales AS s
    ON s.stor_id = st.stor_id
GROUP BY
    st.stor_id,
    st.stor_name,
    st.city,
    st.state
ORDER BY
    qty_sold DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["pubs", "sales", "aggregate"],
        "must_contain": ["dbo.sales", "dbo.stores", "qty"],
        "required_objects": ["dbo.sales", "dbo.stores"],
        "rubric": "This schema snapshot details dbo.sales and dbo.stores. Sum qty. Do not invent a price column on sales.",
    },
)

add(
    id="syn-intent-nj-006",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- customer order history for ALFKI",
    gold="EXEC dbo.CustOrderHist @CustomerID = N'ALFKI';",
    eval={
        "difficulty": "easy",
        "tags": ["procedure", "exec"],
        "must_contain": ["CustOrderHist", "ALFKI"],
        "required_objects": ["dbo.CustOrderHist"],
        "rubric": "Call the listed procedure with its listed parameter. Do not rewrite it as an ad-hoc Orders query unless asked.",
    },
)

add(
    id="syn-intent-nj-007",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- category sales for 1997",
    gold="""SELECT
    CategoryName,
    CategorySales
FROM dbo.[Category Sales for 1997]
ORDER BY
    CategorySales DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["view", "quoted-identifier"],
        "must_contain": ["Category Sales for 1997"],
        "required_objects": ["dbo.Category Sales for 1997"],
        "rubric": "Use the detailed view. Quote the space in the name. Do not rebuild from names-only Order Details.",
    },
)

add(
    id="syn-intent-nj-008",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- customers with no orders",
    gold="""SELECT
    c.CustomerID,
    c.CompanyName,
    c.ContactName,
    c.City,
    c.Country
FROM dbo.Customers AS c
WHERE NOT EXISTS (
    SELECT 1
    FROM dbo.Orders AS o
    WHERE o.CustomerID = c.CustomerID
)
ORDER BY
    c.CompanyName;""",
    eval={
        "difficulty": "medium",
        "tags": ["anti-join", "northwind"],
        "must_contain": ["dbo.Customers", "dbo.Orders"],
        "required_objects": ["dbo.Customers", "dbo.Orders"],
        "rubric": "Customers without Orders. FK CustomerID is listed.",
    },
)

add(
    id="syn-intent-nj-009",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- all columns from the copy column name escape test table",
    gold="""SELECT
    Id,
    NormalName,
    [First Name],
    [Last Name],
    [Order],
    [Select],
    [From],
    [Group],
    [Table],
    [Column-With-Dash],
    [Column.With.Dot],
    [Column/With/Slash]
FROM dbo.CopyColumnNameEscapeTest;""",
    eval={
        "difficulty": "hard",
        "tags": ["quoting", "identifiers"],
        "must_contain": ["CopyColumnNameEscapeTest", "[Order]", "[Select]"],
        "required_objects": ["dbo.CopyColumnNameEscapeTest"],
        "rubric": "Quote reserved words, spaces, dots, dashes, slashes. Do not rename columns.",
    },
)

add(
    id="syn-intent-nj-010",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- run ten most expensive products",
    gold="EXEC dbo.[Ten Most Expensive Products];",
    eval={
        "difficulty": "easy",
        "tags": ["procedure", "names-only", "exec"],
        "must_contain": ["Ten Most Expensive Products"],
        "required_objects": ["dbo.Ten Most Expensive Products"],
        "rubric": "Names-only routines may be EXECed by name. Do not invent parameters.",
    },
)

add(
    id="syn-intent-nj-011",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- list discounts",
    gold="""SELECT
    discounttype,
    stor_id,
    lowqty,
    highqty,
    discount
FROM dbo.discounts
ORDER BY
    discounttype,
    stor_id;""",
    eval={
        "difficulty": "easy",
        "tags": ["pubs", "discounts"],
        "must_contain": ["dbo.discounts"],
        "required_objects": ["dbo.discounts"],
        "rubric": "Select listed dbo.discounts columns.",
    },
)

add(
    id="syn-intent-nj-012",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- current product list",
    gold="""SELECT
    ProductID,
    ProductName
FROM dbo.[Current Product List]
ORDER BY
    ProductName;""",
    eval={
        "difficulty": "easy",
        "tags": ["view", "quoted-identifier"],
        "must_contain": ["Current Product List"],
        "required_objects": ["dbo.Current Product List"],
        "rubric": "Use the detailed view with quoted name. Only ProductID and ProductName are listed.",
    },
)

add(
    id="syn-intent-nj-013",
    completion_category="intent",
    schema_id="ninjadb_b",
    inferred_system_query=False,
    user_comment="-- employees who live in London",
    gold="""SELECT
    EmployeeID,
    LastName,
    FirstName,
    Title,
    HireDate,
    Address,
    City,
    Region,
    PostalCode,
    Country
FROM dbo.Employees
WHERE City = N'London'
ORDER BY
    LastName,
    FirstName;""",
    eval={
        "difficulty": "easy",
        "tags": ["northwind", "employees"],
        "must_contain": ["dbo.Employees", "London"],
        "required_objects": ["dbo.Employees"],
        "rubric": "Employees is detailed on ninjadb_b. Filter City. Do not invent Photo or Notes columns (not listed).",
    },
)

add(
    id="syn-intent-nj-014",
    completion_category="intent",
    schema_id="ninjadb_b",
    inferred_system_query=False,
    user_comment="-- stores and their cities",
    gold="""SELECT
    stor_id,
    stor_name,
    stor_address,
    city,
    state,
    zip
FROM dbo.stores
ORDER BY
    state,
    city,
    stor_name;""",
    eval={
        "difficulty": "easy",
        "tags": ["pubs", "stores"],
        "must_contain": ["dbo.stores"],
        "required_objects": ["dbo.stores"],
        "rubric": "List dbo.stores with listed columns.",
    },
)

add(
    id="syn-intent-nj-015",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- title authors and royalty percent",
    gold="""SELECT
    a.au_id,
    a.au_lname,
    a.au_fname,
    ta.title_id,
    ta.au_ord,
    ta.royaltyper
FROM dbo.authors AS a
INNER JOIN dbo.titleauthor AS ta
    ON ta.au_id = a.au_id
ORDER BY
    a.au_lname,
    ta.au_ord;""",
    eval={
        "difficulty": "medium",
        "tags": ["pubs", "join"],
        "must_contain": ["dbo.titleauthor", "dbo.authors", "royaltyper"],
        "must_not_contain": ["dbo.titles"],
        "required_objects": ["dbo.authors", "dbo.titleauthor"],
        "rubric": "Join authors to titleauthor. dbo.titles is names-only so do not select titles.title or similar columns.",
    },
)

add(
    id="syn-intent-nj-016",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- products at or below reorder level",
    gold="""SELECT
    ProductID,
    ProductName,
    UnitsInStock,
    UnitsOnOrder,
    ReorderLevel,
    Discontinued
FROM dbo.Products
WHERE UnitsInStock <= ReorderLevel
  AND Discontinued = 0
ORDER BY
    UnitsInStock,
    ProductName;""",
    eval={
        "difficulty": "easy",
        "tags": ["products", "predicate"],
        "must_contain": ["dbo.Products", "ReorderLevel", "UnitsInStock"],
        "required_objects": ["dbo.Products"],
        "rubric": "Use listed stock/reorder columns on dbo.Products (detailed in ninjadb_a).",
    },
)

add(
    id="syn-intent-nj-017",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- customers in the WA region",
    gold="""SELECT
    CustomerID,
    CompanyName,
    ContactName,
    City,
    Region,
    PostalCode,
    Country
FROM dbo.Customers
WHERE Region = N'WA'
ORDER BY
    City,
    CompanyName;""",
    eval={
        "difficulty": "easy",
        "tags": ["filter", "region"],
        "must_contain": ["dbo.Customers", "Region"],
        "required_objects": ["dbo.Customers"],
        "rubric": "Customers.Region is listed. Do not use the names-only dbo.Region table columns.",
    },
)

add(
    id="syn-intent-nj-018",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- orders with customer company name and freight",
    gold="""SELECT
    o.OrderID,
    o.OrderDate,
    o.RequiredDate,
    o.ShippedDate,
    o.Freight,
    o.ShipCity,
    o.ShipRegion,
    c.CustomerID,
    c.CompanyName,
    c.Country
FROM dbo.Orders AS o
INNER JOIN dbo.Customers AS c
    ON c.CustomerID = o.CustomerID
ORDER BY
    o.OrderDate DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["join", "orders"],
        "must_contain": ["dbo.Orders", "dbo.Customers", "Freight"],
        "required_objects": ["dbo.Orders", "dbo.Customers"],
        "rubric": "Join Orders to Customers on CustomerID. Stay within listed columns.",
    },
)

add(
    id="syn-intent-nj-019",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- explore order details",
    gold="SELECT *\nFROM dbo.[Order Details];",
    eval={
        "difficulty": "medium",
        "tags": ["names-only", "select-star", "quoted-identifier"],
        "must_contain": ["Order Details"],
        "must_not_contain": ["Quantity", "UnitPrice", "Discount"],
        "required_objects": ["dbo.Order Details"],
        "rubric": "Order Details is names-only. SELECT * (or EXEC-style exploration) is allowed. Projecting Quantity/UnitPrice must be empty instead — those columns are unknown.",
    },
)

add(
    id="syn-intent-nj-020",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- employee sales by country for 1997",
    gold="""EXEC dbo.[Employee Sales by Country]
    @Beginning_Date = '19970101',
    @Ending_Date = '19971231';""",
    eval={
        "difficulty": "medium",
        "tags": ["procedure", "parameters", "quoted-identifier"],
        "must_contain": ["Employee Sales by Country", "Beginning_Date", "Ending_Date"],
        "required_objects": ["dbo.Employee Sales by Country"],
        "rubric": "Call the listed procedure with both listed datetime parameters.",
    },
)

add(
    id="syn-intent-nj-021",
    completion_category="intent",
    schema_id="ninjadb_b",
    inferred_system_query=False,
    user_comment="-- authors and publisher cities from the titles full view",
    gold="""SELECT
    au_id,
    au_lname,
    au_fname,
    author_city,
    author_state,
    pub_id,
    publisher_city,
    publisher_state
FROM dbo.titles_full_view
ORDER BY
    au_lname,
    au_fname;""",
    eval={
        "difficulty": "easy",
        "tags": ["view", "pubs"],
        "must_contain": ["dbo.titles_full_view", "publisher_city"],
        "must_not_contain": ["dbo.publishers"],
        "required_objects": ["dbo.titles_full_view"],
        "rubric": "publishers is names-only; titles_full_view is detailed and already exposes publisher_city.",
    },
)

add(
    id="syn-intent-nj-022",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- alphabetical list of products",
    gold="""SELECT
    ProductID,
    ProductName,
    CategoryName,
    UnitPrice,
    UnitsInStock,
    Discontinued
FROM dbo.[Alphabetical list of products]
ORDER BY
    ProductName;""",
    eval={
        "difficulty": "easy",
        "tags": ["view", "quoted-identifier"],
        "must_contain": ["Alphabetical list of products"],
        "required_objects": ["dbo.Alphabetical list of products"],
        "rubric": "Use the detailed view. Quote the name.",
    },
)

add(
    id="syn-intent-nj-023",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- freight totals by ship city",
    gold="""SELECT
    ShipCity,
    ShipRegion,
    COUNT(*) AS order_count,
    SUM(Freight) AS total_freight
FROM dbo.Orders
GROUP BY
    ShipCity,
    ShipRegion
ORDER BY
    total_freight DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["aggregate", "orders"],
        "must_contain": ["dbo.Orders", "Freight", "ShipCity"],
        "required_objects": ["dbo.Orders"],
        "rubric": "Aggregate listed Orders.Freight by ShipCity. Line-item sales would need Order Details columns, which are unknown.",
    },
)

add(
    id="syn-intent-nj-024",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- sales by customer region using freight as the measure",
    gold="""SELECT
    c.Region,
    COUNT(o.OrderID) AS order_count,
    SUM(o.Freight) AS freight_total
FROM dbo.Customers AS c
LEFT JOIN dbo.Orders AS o
    ON o.CustomerID = c.CustomerID
GROUP BY
    c.Region
ORDER BY
    freight_total DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["sales-by-region", "proxy-measure"],
        "must_contain": ["dbo.Customers", "Region", "dbo.Orders"],
        "must_not_contain": ["dbo.[Order Details]", "ExtendedPrice"],
        "required_objects": ["dbo.Customers", "dbo.Orders"],
        "rubric": "Order Details is names-only so true sales amount is unknown. Freight + Customers.Region is the grounded measure.",
    },
)

add(
    id="syn-intent-nj-025",
    completion_category="intent",
    schema_id="ninjadb_b",
    inferred_system_query=False,
    user_comment="-- sales by store state",
    gold="""SELECT
    st.state,
    SUM(s.qty) AS qty_sold,
    COUNT(*) AS sale_rows
FROM dbo.sales AS s
INNER JOIN dbo.stores AS st
    ON st.stor_id = s.stor_id
GROUP BY
    st.state
ORDER BY
    qty_sold DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["pubs", "sales-by-region"],
        "must_contain": ["dbo.sales", "dbo.stores", "state"],
        "required_objects": ["dbo.sales", "dbo.stores"],
        "rubric": "On ninjadb_b, sales.qty and stores.state are detailed. That is the grounded 'sales by region' for pubs.",
    },
)

add(
    id="syn-intent-nj-026",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- order extended prices by product",
    gold="""SELECT
    ProductID,
    ProductName,
    SUM(ExtendedPrice) AS extended_sales,
    SUM(Quantity) AS qty_sold
FROM dbo.[Order Details Extended]
GROUP BY
    ProductID,
    ProductName
ORDER BY
    extended_sales DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["view", "aggregate"],
        "must_contain": ["Order Details Extended", "ExtendedPrice"],
        "required_objects": ["dbo.Order Details Extended"],
        "rubric": "The detailed view exposes Quantity and ExtendedPrice. Use it instead of the names-only Order Details table.",
    },
)

add(
    id="syn-intent-nj-027",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- orders for customer ALFKI via the stored procedure",
    gold="EXEC dbo.CustOrdersOrders @CustomerID = N'ALFKI';",
    eval={
        "difficulty": "easy",
        "tags": ["procedure"],
        "must_contain": ["CustOrdersOrders", "ALFKI"],
        "required_objects": ["dbo.CustOrdersOrders"],
        "rubric": "Use the listed procedure and parameter.",
    },
)

add(
    id="syn-intent-nj-028",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- authors at 100 percent royalty using byroyalty",
    gold="EXEC dbo.byroyalty @percentage = 100;",
    eval={
        "difficulty": "easy",
        "tags": ["procedure"],
        "must_contain": ["byroyalty", "percentage"],
        "required_objects": ["dbo.byroyalty"],
        "rubric": "dbo.byroyalty takes @percentage int.",
    },
)

add(
    id="syn-intent-nj-029",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- run reptq3 for business titles between 10 and 30 dollars",
    gold="""EXEC dbo.reptq3
    @lolimit = 10,
    @hilimit = 30,
    @type = 'business';""",
    eval={
        "difficulty": "medium",
        "tags": ["procedure", "multi-param"],
        "must_contain": ["reptq3", "lolimit", "hilimit", "type"],
        "required_objects": ["dbo.reptq3"],
        "rubric": "All three listed parameters must be supplied.",
    },
)

add(
    id="syn-intent-nj-030",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- invoices with ship region",
    gold="""SELECT
    CustomerID,
    CustomerName,
    ShipCity,
    ShipRegion,
    ShipCountry,
    City,
    Region,
    PostalCode
FROM dbo.Invoices
ORDER BY
    ShipCountry,
    ShipRegion,
    CustomerName;""",
    eval={
        "difficulty": "easy",
        "tags": ["view", "invoices"],
        "must_contain": ["dbo.Invoices", "ShipRegion"],
        "required_objects": ["dbo.Invoices"],
        "rubric": "Use listed Invoices columns only. This snapshot does not include an amount column on Invoices.",
    },
)

add(
    id="syn-intent-nj-031",
    completion_category="intent",
    schema_id="ninjadb_b",
    inferred_system_query=False,
    user_comment="-- suppliers in the USA",
    gold="""SELECT
    SupplierID,
    CompanyName,
    ContactName,
    City,
    Region,
    PostalCode,
    Country,
    Phone
FROM dbo.Suppliers
WHERE Country = N'USA'
ORDER BY
    CompanyName;""",
    eval={
        "difficulty": "easy",
        "tags": ["suppliers"],
        "must_contain": ["dbo.Suppliers", "USA"],
        "required_objects": ["dbo.Suppliers"],
        "rubric": "Suppliers is detailed on ninjadb_b. Filter Country.",
    },
)

add(
    id="syn-intent-nj-032",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- customers and suppliers by city",
    gold="""SELECT
    City,
    CompanyName,
    ContactName,
    Relationship
FROM dbo.[Customer and Suppliers by City]
ORDER BY
    City,
    Relationship,
    CompanyName;""",
    eval={
        "difficulty": "easy",
        "tags": ["view", "quoted-identifier"],
        "must_contain": ["Customer and Suppliers by City"],
        "required_objects": ["dbo.Customer and Suppliers by City"],
        "rubric": "Quoted detailed view. Only listed columns.",
    },
)

add(
    id="syn-intent-nj-033",
    completion_category="intent",
    schema_id="ninjadb_b",
    inferred_system_query=False,
    user_comment="-- employees hired in 1993 or later",
    gold="""SELECT
    EmployeeID,
    LastName,
    FirstName,
    Title,
    HireDate,
    City,
    Country
FROM dbo.Employees
WHERE HireDate >= '19930101'
ORDER BY
    HireDate,
    LastName;""",
    eval={
        "difficulty": "easy",
        "tags": ["employees", "date-filter"],
        "must_contain": ["dbo.Employees", "HireDate"],
        "required_objects": ["dbo.Employees"],
        "rubric": "Filter Employees.HireDate. Notes/Photo are not listed.",
    },
)

add(
    id="syn-intent-nj-034",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- secret data from AE_Repro",
    gold="""SELECT
    Id,
    SecretData
FROM dbo.AE_Repro
ORDER BY
    Id;""",
    eval={
        "difficulty": "easy",
        "tags": ["select", "small-table"],
        "must_contain": ["dbo.AE_Repro", "SecretData"],
        "required_objects": ["dbo.AE_Repro"],
        "rubric": "Two listed columns only.",
    },
)

add(
    id="syn-intent-nj-035",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- order details for order 10248",
    gold="EXEC dbo.CustOrdersDetail @OrderID = 10248;",
    eval={
        "difficulty": "medium",
        "tags": ["procedure", "names-only-avoidance"],
        "must_contain": ["CustOrdersDetail", "10248"],
        "must_not_contain": ["dbo.[Order Details]"],
        "required_objects": ["dbo.CustOrdersDetail"],
        "rubric": "Need line details but the table is names-only. The listed procedure CustOrdersDetail(@OrderID) is the grounded path.",
    },
)

# ===========================================================================
# System / DMV intent
# ===========================================================================

add(
    id="syn-intent-sys-001",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- currently running requests with sql text",
    gold="""SELECT
    r.session_id,
    r.status,
    r.command,
    r.database_id,
    r.blocking_session_id,
    r.wait_type,
    r.wait_time,
    r.cpu_time,
    r.total_elapsed_time,
    st.text AS query_text
FROM sys.dm_exec_requests AS r
CROSS APPLY sys.dm_exec_sql_text(r.sql_handle) AS st
WHERE r.session_id <> @@SPID
ORDER BY
    r.total_elapsed_time DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["dmv", "requests"],
        "must_contain": ["sys.dm_exec_requests", "sys.dm_exec_sql_text"],
        "required_objects": ["sys.dm_exec_requests", "sys.dm_exec_sql_text"],
        "rubric": "Running requests plus sql_text via CROSS APPLY. Exclude @@SPID.",
    },
)

add(
    id="syn-intent-sys-002",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- who is blocking whom",
    gold="""SELECT
    r.session_id AS blocked_session_id,
    r.blocking_session_id,
    r.wait_type,
    r.wait_time,
    r.status,
    r.command,
    DB_NAME(r.database_id) AS database_name,
    st.text AS blocked_query_text
FROM sys.dm_exec_requests AS r
CROSS APPLY sys.dm_exec_sql_text(r.sql_handle) AS st
WHERE r.blocking_session_id <> 0
ORDER BY
    r.wait_time DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["dmv", "blocking"],
        "must_contain": ["blocking_session_id", "sys.dm_exec_requests"],
        "required_objects": ["sys.dm_exec_requests"],
        "rubric": "Filter blocking_session_id <> 0 on dm_exec_requests.",
    },
)

add(
    id="syn-intent-sys-003",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- top waits on this server",
    gold="""SELECT TOP (20)
    wait_type,
    waiting_tasks_count,
    wait_time_ms,
    max_wait_time_ms,
    signal_wait_time_ms
FROM sys.dm_os_wait_stats
WHERE wait_type NOT LIKE N'SLEEP%'
  AND wait_type NOT LIKE N'BROKER%'
ORDER BY
    wait_time_ms DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["dmv", "waits", "onprem"],
        "must_contain": ["sys.dm_os_wait_stats", "wait_time_ms"],
        "required_objects": ["sys.dm_os_wait_stats"],
        "rubric": "sys.dm_os_wait_stats is listed on this on-prem catalog.",
    },
)

add(
    id="syn-intent-sys-004",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=True,
    user_comment="-- missing index suggestions",
    gold="""SELECT
    mid.database_id,
    DB_NAME(mid.database_id) AS database_name,
    mid.object_id,
    mid.equality_columns,
    mid.inequality_columns,
    mid.included_columns,
    mid.statement
FROM sys.dm_db_missing_index_details AS mid
WHERE mid.database_id = DB_ID()
ORDER BY
    mid.object_id;""",
    eval={
        "difficulty": "medium",
        "tags": ["dmv", "indexes"],
        "must_contain": ["sys.dm_db_missing_index_details"],
        "required_objects": ["sys.dm_db_missing_index_details"],
        "rubric": "Use listed missing_index_details columns. Do not invent dm_db_missing_index_group_stats (not listed).",
    },
)

add(
    id="syn-intent-sys-005",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- databases that are online",
    gold="""SELECT
    database_id,
    name,
    state_desc,
    compatibility_level,
    collation_name,
    recovery_model_desc,
    create_date
FROM sys.databases
WHERE state_desc = N'ONLINE'
ORDER BY
    name;""",
    eval={
        "difficulty": "easy",
        "tags": ["catalog", "sys.databases"],
        "must_contain": ["sys.databases", "ONLINE"],
        "required_objects": ["sys.databases"],
        "rubric": "Filter sys.databases.state_desc.",
    },
)

add(
    id="syn-intent-sys-006",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- index fragmentation in this database",
    gold="""SELECT
    object_id,
    index_id,
    partition_number,
    index_type_desc,
    alloc_unit_type_desc,
    avg_fragmentation_in_percent,
    page_count
FROM sys.dm_db_index_physical_stats(DB_ID(), NULL, NULL, NULL, N'LIMITED')
WHERE avg_fragmentation_in_percent > 10
  AND page_count > 100
ORDER BY
    avg_fragmentation_in_percent DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["tvf", "fragmentation"],
        "must_contain": ["dm_db_index_physical_stats", "avg_fragmentation_in_percent"],
        "required_objects": ["sys.dm_db_index_physical_stats"],
        "rubric": "dm_db_index_physical_stats is a TVF. Call it with DB_ID(). Do not select it like a base table without arguments.",
    },
)

add(
    id="syn-intent-sys-007",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- top 10 queries by cpu",
    gold="""SELECT TOP (10)
    qs.execution_count,
    qs.total_worker_time,
    qs.total_elapsed_time,
    qs.total_logical_reads,
    qs.last_execution_time,
    qs.total_worker_time / NULLIF(qs.execution_count, 0) AS avg_worker_us,
    SUBSTRING(st.text, 1, 200) AS query_text
FROM sys.dm_exec_query_stats AS qs
CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) AS st
ORDER BY
    qs.total_worker_time DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["dmv", "cpu"],
        "must_contain": ["sys.dm_exec_query_stats", "total_worker_time"],
        "required_objects": ["sys.dm_exec_query_stats", "sys.dm_exec_sql_text"],
        "rubric": "Top CPU from query_stats.total_worker_time. Not Query Store.",
    },
)

add(
    id="syn-intent-sys-008",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- disabled sql logins",
    gold="""SELECT
    principal_id,
    name,
    is_disabled,
    is_policy_checked,
    is_expiration_checked,
    default_database_name
FROM sys.sql_logins
WHERE is_disabled = 1
ORDER BY
    name;""",
    eval={
        "difficulty": "easy",
        "tags": ["security", "logins", "onprem"],
        "must_contain": ["sys.sql_logins", "is_disabled"],
        "required_objects": ["sys.sql_logins"],
        "rubric": "sys.sql_logins is listed on this on-prem catalog.",
    },
)

add(
    id="syn-intent-sys-009",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- sessions using the most memory",
    gold="""SELECT TOP (20)
    session_id,
    login_name,
    host_name,
    program_name,
    status,
    cpu_time,
    memory_usage,
    reads,
    writes,
    last_request_end_time
FROM sys.dm_exec_sessions
WHERE session_id > 50
ORDER BY
    memory_usage DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["dmv", "sessions"],
        "must_contain": ["sys.dm_exec_sessions", "memory_usage"],
        "required_objects": ["sys.dm_exec_sessions"],
        "rubric": "Order dm_exec_sessions by listed memory_usage.",
    },
)

add(
    id="syn-intent-sys-010",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=True,
    user_comment="-- indexes that have never been used",
    gold="""SELECT
    i.object_id,
    OBJECT_NAME(i.object_id) AS table_name,
    i.name AS index_name,
    i.type_desc,
    s.user_seeks,
    s.user_scans,
    s.user_lookups,
    s.user_updates
FROM sys.indexes AS i
LEFT JOIN sys.dm_db_index_usage_stats AS s
    ON s.object_id = i.object_id
    AND s.index_id = i.index_id
    AND s.database_id = DB_ID()
WHERE i.is_primary_key = 0
  AND i.type_desc <> N'HEAP'
  AND ISNULL(s.user_seeks, 0) = 0
  AND ISNULL(s.user_scans, 0) = 0
  AND ISNULL(s.user_lookups, 0) = 0
ORDER BY
    s.user_updates DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["indexes", "usage-stats"],
        "must_contain": ["sys.dm_db_index_usage_stats", "sys.indexes"],
        "required_objects": ["sys.dm_db_index_usage_stats", "sys.indexes"],
        "rubric": "Join listed indexes to usage stats. Do not invent operational_stats DMVs.",
    },
)

add(
    id="syn-intent-sys-011",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=True,
    user_comment="-- foreign keys in this database",
    gold="""SELECT
    fk.name AS foreign_key_name,
    OBJECT_NAME(fk.parent_object_id) AS parent_table,
    OBJECT_NAME(fk.referenced_object_id) AS referenced_table,
    fk.delete_referential_action_desc,
    fk.update_referential_action_desc
FROM sys.foreign_keys AS fk
ORDER BY
    parent_table,
    foreign_key_name;""",
    eval={
        "difficulty": "easy",
        "tags": ["catalog", "fk"],
        "must_contain": ["sys.foreign_keys"],
        "required_objects": ["sys.foreign_keys"],
        "rubric": "List sys.foreign_keys. Joining foreign_key_columns is optional.",
    },
)

add(
    id="syn-intent-sys-012",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- wait for 10 seconds",
    gold="WAITFOR DELAY '00:00:10';",
    eval={
        "difficulty": "easy",
        "tags": ["waitfor"],
        "must_contain": ["WAITFOR DELAY", "00:00:10"],
        "rubric": "WAITFOR DELAY ten seconds.",
    },
)

add(
    id="syn-intent-sys-013",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- memory clerks by pages",
    gold="""SELECT
    type,
    name,
    memory_node_id,
    pages_kb,
    virtual_memory_reserved_kb,
    virtual_memory_committed_kb
FROM sys.dm_os_memory_clerks
ORDER BY
    pages_kb DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["memory", "onprem"],
        "must_contain": ["sys.dm_os_memory_clerks", "pages_kb"],
        "required_objects": ["sys.dm_os_memory_clerks"],
        "rubric": "Listed only on the on-prem catalog.",
    },
)

add(
    id="syn-intent-sys-014",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- server endpoints",
    gold="""SELECT
    endpoint_id,
    name,
    protocol_desc,
    type_desc,
    state_desc,
    is_admin_endpoint
FROM sys.endpoints
ORDER BY
    name;""",
    eval={
        "difficulty": "easy",
        "tags": ["endpoints", "onprem"],
        "must_contain": ["sys.endpoints"],
        "required_objects": ["sys.endpoints"],
        "rubric": "sys.endpoints is listed on on-prem master.",
    },
)

add(
    id="syn-intent-sys-015",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- disabled server principals",
    gold="""SELECT
    principal_id,
    name,
    type_desc,
    is_disabled,
    create_date,
    default_database_name
FROM sys.server_principals
WHERE is_disabled = 1
ORDER BY
    name;""",
    eval={
        "difficulty": "easy",
        "tags": ["security", "onprem"],
        "must_contain": ["sys.server_principals", "is_disabled"],
        "required_objects": ["sys.server_principals"],
        "rubric": "Filter listed is_disabled on server_principals.",
    },
)

add(
    id="syn-intent-sys-016",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=True,
    user_comment="-- show the estimated query plan for cached plans",
    gold="""SELECT
    qs.plan_handle,
    qs.execution_count,
    qs.total_worker_time,
    qp.query_plan
FROM sys.dm_exec_query_stats AS qs
CROSS APPLY sys.dm_exec_query_plan(qs.plan_handle) AS qp
ORDER BY
    qs.total_worker_time DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["query-plan"],
        "must_contain": ["sys.dm_exec_query_plan", "plan_handle"],
        "required_objects": ["sys.dm_exec_query_plan", "sys.dm_exec_query_stats"],
        "rubric": "dm_exec_query_plan is a TVF on plan_handle.",
    },
)

add(
    id="syn-intent-sys-017",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- databases and compatibility level",
    gold="""SELECT
    name,
    compatibility_level,
    collation_name,
    recovery_model_desc,
    state_desc,
    create_date
FROM sys.databases
ORDER BY
    name;""",
    eval={
        "difficulty": "easy",
        "tags": ["sys.databases"],
        "must_contain": ["sys.databases", "compatibility_level"],
        "required_objects": ["sys.databases"],
        "rubric": "Project listed sys.databases columns.",
    },
)

add(
    id="syn-intent-sys-018",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- sleeping sessions",
    gold="""SELECT
    session_id,
    login_name,
    host_name,
    program_name,
    status,
    cpu_time,
    memory_usage,
    last_request_end_time
FROM sys.dm_exec_sessions
WHERE status = N'sleeping'
  AND session_id > 50
ORDER BY
    last_request_end_time;""",
    eval={
        "difficulty": "easy",
        "tags": ["sessions"],
        "must_contain": ["sys.dm_exec_sessions", "sleeping"],
        "required_objects": ["sys.dm_exec_sessions"],
        "rubric": "Filter dm_exec_sessions.status.",
    },
)

add(
    id="syn-intent-sys-019",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=True,
    user_comment="-- procedures and their parameters",
    gold="""SELECT
    s.name AS schema_name,
    p.name AS procedure_name,
    pr.parameter_id,
    pr.name AS parameter_name,
    t.name AS type_name,
    pr.max_length,
    pr.is_output
FROM sys.procedures AS p
INNER JOIN sys.schemas AS s
    ON s.schema_id = p.schema_id
LEFT JOIN sys.parameters AS pr
    ON pr.object_id = p.object_id
LEFT JOIN sys.types AS t
    ON t.user_type_id = pr.user_type_id
ORDER BY
    s.name,
    p.name,
    pr.parameter_id;""",
    eval={
        "difficulty": "medium",
        "tags": ["catalog", "procedures"],
        "must_contain": ["sys.procedures", "sys.parameters"],
        "required_objects": ["sys.procedures", "sys.parameters"],
        "rubric": "Join listed catalog views. FitnessApp has no user routines, so this still must be catalog SQL, not invented procedures.",
    },
)

add(
    id="syn-intent-sys-020",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- os performance counters",
    gold="""SELECT
    object_name,
    counter_name,
    instance_name,
    cntr_value,
    cntr_type
FROM sys.dm_os_performance_counters
ORDER BY
    object_name,
    counter_name;""",
    eval={
        "difficulty": "easy",
        "tags": ["perf-counters", "onprem"],
        "must_contain": ["sys.dm_os_performance_counters"],
        "required_objects": ["sys.dm_os_performance_counters"],
        "rubric": "Listed on on-prem only.",
    },
)

add(
    id="syn-intent-sys-021",
    completion_category="intent",
    schema_id="azure_master",
    inferred_system_query=True,
    user_comment="-- active sessions and their host",
    gold="""SELECT
    s.session_id,
    s.login_name,
    s.host_name,
    s.program_name,
    s.status,
    s.cpu_time,
    s.memory_usage,
    c.connect_time,
    c.client_net_address
FROM sys.dm_exec_sessions AS s
LEFT JOIN sys.dm_exec_connections AS c
    ON c.session_id = s.session_id
WHERE s.session_id > 50
ORDER BY
    s.cpu_time DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["azure", "sessions"],
        "must_contain": ["sys.dm_exec_sessions"],
        "required_objects": ["sys.dm_exec_sessions"],
        "rubric": "Azure master lists sessions and connections. Do not use sql_logins or master_files.",
    },
)

add(
    id="syn-intent-sys-022",
    completion_category="intent",
    schema_id="azure_master",
    inferred_system_query=False,
    user_comment="-- current dac username",
    gold="SELECT dbo.fn_sysdac_get_currentusername() AS current_username;",
    eval={
        "difficulty": "easy",
        "tags": ["scalar-function", "azure"],
        "must_contain": ["fn_sysdac_get_currentusername"],
        "required_objects": ["dbo.fn_sysdac_get_currentusername"],
        "rubric": "Only user routine on azure master is this scalar function.",
    },
)

# ===========================================================================
# Empty-string / trap cases
# ===========================================================================

add(
    id="syn-empty-001",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- sales by region",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "trap", "missing-schema"],
        "expect_empty": True,
        "must_not_contain": ["SELECT", "dbo.Orders", "Region", "```"],
        "rubric": "FitnessApp has no sales or region objects. Empty string, not a guessed Northwind query.",
    },
)

add(
    id="syn-empty-002",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- employees hired last year",
    gold="",
    eval={
        "difficulty": "easy",
        "tags": ["empty-policy", "trap"],
        "expect_empty": True,
        "must_not_contain": ["SELECT", "dbo.Employees"],
        "rubric": "No Employees table in FitnessApp.",
    },
)

add(
    id="syn-empty-003",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- order details for customer ALFKI",
    gold="",
    eval={
        "difficulty": "easy",
        "tags": ["empty-policy", "trap"],
        "expect_empty": True,
        "must_not_contain": ["SELECT", "ALFKI", "dbo.Orders"],
        "rubric": "Northwind objects are not in this schema.",
    },
)

add(
    id="syn-empty-004",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- quantity and unit price from order details",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["empty-policy", "names-only", "trap"],
        "expect_empty": True,
        "must_not_contain": ["Quantity", "UnitPrice", "SELECT"],
        "rubric": "dbo.[Order Details] is names-only. Requesting specific columns must return empty. (Use the detailed Order Details Extended view only if the prompt names that view.)",
    },
)

add(
    id="syn-empty-005",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- titles with price and pub_id",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "names-only"],
        "expect_empty": True,
        "must_not_contain": ["dbo.titles", "price", "SELECT"],
        "rubric": "dbo.titles is names-only. price/pub_id are unknown.",
    },
)

add(
    id="syn-empty-006",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- publisher names and cities from publishers",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "names-only"],
        "expect_empty": True,
        "must_not_contain": ["pub_name", "SELECT"],
        "rubric": "dbo.publishers is names-only. Do not invent pub_name. titles_full_view is a different object and should not be substituted unless the prompt allows it.",
    },
)

add(
    id="syn-empty-007",
    completion_category="intent",
    schema_id="azure_master",
    inferred_system_query=True,
    user_comment="-- top waits on this server",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["empty-policy", "azure", "unlisted-dmv"],
        "expect_empty": True,
        "must_not_contain": ["sys.dm_os_wait_stats", "SELECT"],
        "rubric": "sys.dm_os_wait_stats is not listed on the Azure master snapshot.",
    },
)

add(
    id="syn-empty-008",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- what is the size of the database",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["empty-policy", "azure", "master_files"],
        "expect_empty": True,
        "must_not_contain": ["sys.master_files", "sys.database_files", "SELECT"],
        "rubric": "Azure ninjadb snapshot does not list sys.master_files or sys.database_files. Empty rather than inventing.",
    },
)

add(
    id="syn-empty-009",
    completion_category="intent",
    schema_id="azure_master",
    inferred_system_query=True,
    user_comment="-- disabled sql logins",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "azure", "logins"],
        "expect_empty": True,
        "must_not_contain": ["sys.sql_logins", "SELECT"],
        "rubric": "sys.sql_logins is not listed on Azure master.",
    },
)

add(
    id="syn-empty-010",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- top queries from query store runtime stats",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "query-store"],
        "expect_empty": True,
        "must_not_contain": ["query_store", "sys.dm_exec_query_stats", "SELECT"],
        "rubric": "Query Store objects are not listed. Do not silently substitute query_stats.",
    },
)

add(
    id="syn-empty-011",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- jobs and job descriptions",
    gold="",
    eval={
        "difficulty": "easy",
        "tags": ["empty-policy"],
        "expect_empty": True,
        "must_not_contain": ["dbo.jobs", "SELECT"],
        "rubric": "No jobs table in FitnessApp.",
    },
)

add(
    id="syn-empty-012",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- job descriptions from jobs",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "names-only"],
        "expect_empty": True,
        "must_not_contain": ["job_desc", "SELECT"],
        "rubric": "dbo.jobs is names-only; job_desc is unknown.",
    },
)

add(
    id="syn-empty-013",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- sales by category from the sales by category view with category name and sales",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["empty-policy", "names-only-view"],
        "expect_empty": True,
        "must_not_contain": ["SELECT", "CategorySales"],
        "rubric": "VIEW NAMES dbo (Sales by Category) is names-only. Requesting specific columns must be empty. Category Sales for 1997 is a different detailed view and should not be silently substituted.",
    },
)

add(
    id="syn-empty-014",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- employee territories with territory description",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "names-only"],
        "expect_empty": True,
        "must_not_contain": ["EmployeeTerritories", "SELECT"],
        "rubric": "EmployeeTerritories/Territories are names-only. TerritoryDescription is unknown.",
    },
)

add(
    id="syn-empty-015",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=True,
    user_comment="-- os wait stats",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "azure", "unlisted-dmv"],
        "expect_empty": True,
        "must_not_contain": ["sys.dm_os_wait_stats", "SELECT"],
        "rubric": "Wait stats DMV is not on the Azure ninjadb catalog list.",
    },
)

add(
    id="syn-empty-016",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=False,
    user_comment="-- meals over 500 calories",
    gold="",
    eval={
        "difficulty": "easy",
        "tags": ["empty-policy", "wrong-database"],
        "expect_empty": True,
        "must_not_contain": ["dbo.Meals", "SELECT"],
        "rubric": "Connected to master with zero user tables. Do not use FitnessApp objects.",
    },
)

add(
    id="syn-empty-017",
    completion_category="intent",
    schema_id="azure_master",
    inferred_system_query=True,
    user_comment="-- database file sizes from master_files",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "azure"],
        "expect_empty": True,
        "must_not_contain": ["sys.master_files", "SELECT"],
        "rubric": "sys.master_files is not listed on Azure master.",
    },
)

add(
    id="syn-empty-018",
    completion_category="intent",
    schema_id="ninjadb_b",
    inferred_system_query=False,
    user_comment="-- product unit prices and discontinued flag",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["empty-policy", "snapshot-specific", "names-only"],
        "expect_empty": True,
        "must_not_contain": ["UnitPrice", "Discontinued", "SELECT"],
        "rubric": "On ninjadb_b, dbo.Products is names-only. UnitPrice is unknown on that snapshot. Do not borrow columns from ninjadb_a.",
    },
)

# ===========================================================================
# Continuation-mode
# ===========================================================================

add(
    id="syn-cont-001",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    user_id,\n    full_name,\n    email\nFROM ",
    line_prefix="FROM ",
    gold="dbo.Users",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "from-clause"],
        "must_contain": ["dbo.Users"],
        "must_not_contain": ["WHERE", ";"],
        "required_objects": ["dbo.Users"],
        "rubric": "Complete FROM with one listed table. One unit, no WHERE, no semicolon.",
    },
)

add(
    id="syn-cont-002",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    u.full_name,\n    p.amount\nFROM dbo.Users AS u\nINNER JOIN ",
    line_prefix="INNER JOIN ",
    gold="dbo.Payments AS p\n    ON p.user_id = u.user_id",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "join", "fk"],
        "must_contain": ["dbo.Payments", "user_id"],
        "must_not_contain": [";", "WHERE"],
        "required_objects": ["dbo.Payments"],
        "rubric": "One JOIN plus ON using the listed FK. Never T.col = T.col tautology.",
    },
)

add(
    id="syn-cont-003",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    meal_name,\n    calories\nFROM dbo.Meals\nWHERE ",
    line_prefix="WHERE ",
    gold="calories > 500",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "where"],
        "must_contain": ["calories"],
        "must_not_contain": [";", "ORDER BY"],
        "rubric": "One predicate on a listed Meals column.",
    },
)

add(
    id="syn-cont-004",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    category,\n    AVG(calories) AS avg_calories\nFROM dbo.Meals\nGROUP BY\n    category\nORDER BY ",
    line_prefix="ORDER BY ",
    gold="avg_calories DESC",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "order-by"],
        "must_contain": ["avg_calories"],
        "must_not_contain": [";"],
        "rubric": "One ORDER BY unit. No semicolon in continuation mode.",
    },
)

add(
    id="syn-cont-005",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment=None,
    statement="SELECT\n    name,\n    state_desc\nFROM sys.",
    line_prefix="FROM sys.",
    gold="databases",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "identifier"],
        "must_contain": ["databases"],
        "must_not_contain": ["sys.", ";"],
        "rubric": "After sys. return the remaining identifier only.",
    },
)

add(
    id="syn-cont-006",
    completion_category="continuation",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT *\nFROM dbo.[Order",
    line_prefix="FROM dbo.[Order",
    gold=" Details]",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "quoted-identifier"],
        "must_contain": ["Details]"],
        "must_not_contain": ["SELECT", ";"],
        "rubric": "Finish the quoted [Order Details] name. Do not add columns.",
    },
)

add(
    id="syn-cont-007",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment=None,
    statement="SELECT\n    qs.execution_count,\n    st.text\nFROM sys.dm_exec_query_stats AS qs\nCROSS APPLY sys.",
    line_prefix="CROSS APPLY sys.",
    gold="dm_exec_sql_text(qs.sql_handle) AS st",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "apply"],
        "must_contain": ["dm_exec_sql_text", "sql_handle"],
        "must_not_contain": [";"],
        "required_objects": ["sys.dm_exec_sql_text"],
        "rubric": "One APPLY unit completing the TVF call. Do not add WHERE.",
    },
)

add(
    id="syn-cont-008",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    meal_name,\n    calories,\n    protein_g\nFROM dbo.Meals\nWHERE calories > 500\n  AND ",
    line_prefix="  AND ",
    gold="protein_g >= 30",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "and-predicate"],
        "must_contain": ["protein_g"],
        "must_not_contain": [";", "ORDER BY"],
        "rubric": "One additional AND predicate on a listed column.",
    },
)

add(
    id="syn-cont-009",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    user_id,\n    full_name",
    line_prefix="    full_name",
    document_suffix="\nFROM dbo.Users\nORDER BY\n    full_name;",
    line_suffix="",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "suffix-aware", "empty-policy"],
        "expect_empty": True,
        "must_not_contain": ["FROM", "Users", "SELECT"],
        "rubric": "Document suffix already has FROM/ORDER BY. No natural extra select-list unit — return empty rather than duplicating FROM.",
    },
)

add(
    id="syn-cont-010",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT *\nFROM dbo.User_",
    line_prefix="FROM dbo.User_",
    gold="Workouts",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "identifier"],
        "must_contain_any": ["Workouts", "Fitness_Goals"],
        "must_not_contain": ["dbo.User_", ";"],
        "rubric": "Complete User_Workouts or User_Fitness_Goals. Do not repeat the prefix.",
    },
)

add(
    id="syn-cont-011",
    completion_category="continuation",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    c.CustomerID,\n    c.CompanyName,\n    o.OrderID\nFROM dbo.Customers AS c\nINNER JOIN dbo.Orders AS o\n    ON ",
    line_prefix="    ON ",
    gold="o.CustomerID = c.CustomerID",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "join-on", "fk"],
        "must_contain": ["CustomerID"],
        "must_not_contain": ["o.CustomerID = o.CustomerID", ";"],
        "rubric": "PK/FK join. Never same-side tautology.",
    },
)

add(
    id="syn-cont-012",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    u.full_name\nFROM dbo.Users AS u\nINNER JOIN dbo.User_Workouts AS uw\n    ON ",
    line_prefix="    ON ",
    gold="uw.user_id = u.user_id",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "fk"],
        "must_contain": ["user_id"],
        "must_not_contain": ["u.user_id = u.user_id"],
        "rubric": "Join on listed FK user_id.",
    },
)

add(
    id="syn-cont-013",
    completion_category="continuation",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    ProductName,\n    UnitPrice\nFROM dbo.Products\nWHERE ",
    line_prefix="WHERE ",
    gold="Discontinued = 0",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "where", "products"],
        "must_contain": ["Discontinued"],
        "must_not_contain": [";"],
        "rubric": "One Products predicate using a listed column.",
    },
)

add(
    id="syn-cont-014",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    meal_id,\n    meal_name\nFROM dbo.",
    line_prefix="FROM dbo.",
    gold="Meals",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "identifier"],
        "must_contain": ["Meals"],
        "must_not_contain": ["dbo.", ";"],
        "rubric": "After dbo. return Meals (or another listed table) without repeating dbo.",
    },
)

add(
    id="syn-cont-015",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    full_name,\n    gender\nFROM dbo.Users\nWHERE ",
    line_prefix="WHERE ",
    gold="gender = N'F'",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "where"],
        "must_contain": ["gender"],
        "must_not_contain": [";"],
        "rubric": "One Users.gender predicate.",
    },
)

add(
    id="syn-cont-016",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment=None,
    statement="SELECT *\nFROM sys.databases\n",
    line_prefix="",
    gold="WHERE state_desc = N'ONLINE'",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "new-clause", "newline"],
        "must_contain": ["WHERE", "state_desc"],
        "must_not_contain": [";", "SELECT"],
        "rubric": "Cursor on a blank line after FROM sys.databases. Next unit is a WHERE clause starting on this line. No semicolon.",
    },
)

add(
    id="syn-cont-017",
    completion_category="continuation",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    au_id,\n    au_lname,\n    au_fname\nFROM dbo.authors\nWHERE ",
    line_prefix="WHERE ",
    gold="state = 'CA'",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "authors"],
        "must_contain": ["state"],
        "must_not_contain": [";"],
        "rubric": "One authors.state predicate.",
    },
)

add(
    id="syn-cont-018",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    u.user_id,\n    u.full_name,\n    COUNT(*) AS workout_count\nFROM dbo.Users AS u\nINNER JOIN dbo.User_Workouts AS uw\n    ON uw.user_id = u.user_id\nGROUP BY ",
    line_prefix="GROUP BY ",
    gold="u.user_id,\n    u.full_name",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "group-by"],
        "must_contain": ["u.user_id", "u.full_name"],
        "must_not_contain": [";", "ORDER BY"],
        "rubric": "GROUP BY the non-aggregated select items only.",
    },
)

add(
    id="syn-cont-019",
    completion_category="continuation",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    p.ProductName,\n    c.CategoryName\nFROM dbo.Products AS p\nINNER JOIN dbo.Categories AS c\n    ON ",
    line_prefix="    ON ",
    gold="c.CategoryID = p.CategoryID",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "fk"],
        "must_contain": ["CategoryID"],
        "must_not_contain": ["p.CategoryID = p.CategoryID"],
        "rubric": "Join Products to Categories on CategoryID.",
    },
)

add(
    id="syn-cont-020",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    challenge_name,\n    start_date,\n    end_date\nFROM dbo.Challenges",
    line_prefix="FROM dbo.Challenges",
    document_suffix="\nWHERE end_date >= CAST(GETDATE() AS date);",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "suffix-aware", "empty-policy"],
        "expect_empty": True,
        "must_not_contain": ["WHERE", "ORDER BY", "JOIN"],
        "rubric": "Suffix already contains WHERE. Do not insert another clause between FROM and that WHERE.",
    },
)

add(
    id="syn-cont-021",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment=None,
    statement="SELECT\n    r.session_id,\n    r.status,\n    r.wait_type\nFROM sys.dm_exec_requests AS r\nWHERE ",
    line_prefix="WHERE ",
    gold="r.session_id <> @@SPID",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "where", "spid"],
        "must_contain_any": ["@@SPID", "session_id"],
        "must_not_contain": [";"],
        "rubric": "One requests predicate, typically exclude self.",
    },
)

add(
    id="syn-cont-022",
    completion_category="continuation",
    schema_id="ninjadb_b",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    s.stor_id,\n    s.qty\nFROM dbo.sales AS s\nINNER JOIN dbo.stores AS st\n    ON ",
    line_prefix="    ON ",
    gold="st.stor_id = s.stor_id",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "fk", "pubs"],
        "must_contain": ["stor_id"],
        "must_not_contain": ["s.stor_id = s.stor_id"],
        "rubric": "sales.stor_id FK to stores.stor_id.",
    },
)

add(
    id="syn-cont-023",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    u.full_name,\n    g.goal_type,\n    g.target_weight\nFROM dbo.Users AS u\nLEFT JOIN dbo.User_Fitness_Goals AS g\n    ON g.user_id = u.user_id\nWHERE ",
    line_prefix="WHERE ",
    gold="g.goal_id IS NULL",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "anti-join"],
        "must_contain": ["IS NULL"],
        "must_not_contain": [";"],
        "rubric": "LEFT JOIN already present; the natural predicate is unmatched goals (g.goal_id IS NULL) or an active-status filter — one unit only.",
    },
)

add(
    id="syn-cont-024",
    completion_category="continuation",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    CompanyName,\n    City,\n    Country\nFROM dbo.Customers",
    line_prefix="FROM dbo.Customers",
    gold="\nWHERE Country = N'USA'",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "new-clause"],
        "must_contain": ["WHERE", "Country"],
        "must_not_contain": [";"],
        "rubric": "Cursor after a complete FROM with no suffix. Next unit is a new WHERE clause, preferably starting with a newline.",
    },
)

add(
    id="syn-cont-025",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    meal_name,\n    calories,\n    protein_g,\n    carbs_g\nFROM dbo.Meals\nORDER BY\n    calories DESC;",
    line_prefix="    calories DESC;",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "complete-statement", "empty-policy"],
        "expect_empty": True,
        "must_not_contain": ["SELECT", "WHERE", "OFFSET"],
        "rubric": "Statement already ended with a semicolon. Return empty rather than appending another query.",
    },
)

# ===========================================================================
# Cursor / format / do-not-repeat-prefix
# ===========================================================================

add(
    id="syn-fmt-001",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- what meals are there",
    statement="-- what meals are there\nSELECT",
    line_prefix="SELECT",
    gold="""
    meal_id,
    meal_name,
    category,
    calories,
    protein_g,
    carbs_g,
    fats_g,
    dietary_preference
FROM dbo.Meals
ORDER BY
    category,
    meal_name;""",
    eval={
        "difficulty": "medium",
        "tags": ["format", "do-not-repeat-select"],
        "must_contain": ["dbo.Meals", "meal_name"],
        "must_not_contain": ["SELECT SELECT", "```"],
        "rubric": "User already typed SELECT. Continue from that keyword — do not emit a second SELECT.",
    },
)

add(
    id="syn-fmt-002",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- list users",
    statement="-- list users\n",
    line_prefix="",
    gold="""SELECT
    user_id,
    full_name,
    email,
    date_of_birth,
    gender,
    height_cm,
    weight_kg,
    created_at
FROM dbo.Users
ORDER BY
    full_name;""",
    eval={
        "difficulty": "easy",
        "tags": ["format", "blank-line"],
        "must_contain": ["SELECT", "dbo.Users"],
        "must_not_contain": ["```", "password_hash"],
        "rubric": "Cursor is on a blank line after the comment. Start at SELECT with no leading prose.",
    },
)

add(
    id="syn-fmt-003",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- customers in France",
    recent="SELECT\n    CustomerID,\n    CompanyName\nFROM dbo.Customers\nWHERE Country = N'Germany';",
    gold="""SELECT
    CustomerID,
    CompanyName,
    ContactName,
    City,
    Country
FROM dbo.Customers
WHERE Country = N'France'
ORDER BY
    City,
    CompanyName;""",
    eval={
        "difficulty": "medium",
        "tags": ["format", "recent-prefix", "do-not-repeat"],
        "must_contain": ["France", "dbo.Customers"],
        "must_not_contain": ["Germany"],
        "rubric": "New comment after a Germany query. Produce a France query. Do not continue or repair the previous statement.",
    },
)

add(
    id="syn-fmt-004",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- what is the best way to resolve this error Incorrect syntax near ','?",
    recent="""WITH bounds AS (
    SELECT
        DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()) - 1, 0) AS last_month_start,
        DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0) AS this_month_start
)
SELECT
    lm.sql_handle,
    lm.plan_handle,
FROM last_month AS lm;""",
    gold="""SELECT
    qs.sql_handle,
    qs.plan_handle,
    qs.execution_count,
    qs.total_elapsed_time,
    SUBSTRING(st.text, 1, 200) AS query_text
FROM sys.dm_exec_query_stats AS qs
CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) AS st
ORDER BY
    qs.total_elapsed_time DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["syntax-error", "do-not-continue-broken-prefix", "trap"],
        "must_contain": ["SELECT"],
        "must_not_contain": ["FROM last_month"],
        "rubric": "User asked how to resolve 'Incorrect syntax near ,'. Do not keep completing the broken CTE fragment. Emit a complete valid statement (or empty if the intended query cannot be recovered). Continuing 'lm.exec_count...' as the origin traces did is a fail.",
    },
)

add(
    id="syn-fmt-005",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- what meals are there",
    gold="""SELECT
    meal_id,
    meal_name,
    category,
    calories,
    protein_g,
    carbs_g,
    fats_g,
    dietary_preference
FROM dbo.Meals
ORDER BY
    category,
    meal_name;""",
    eval={
        "difficulty": "easy",
        "tags": ["format", "no-markdown", "raw-sql"],
        "must_contain": ["dbo.Meals"],
        "must_not_contain": ["```", "Here is", "Sure"],
        "rubric": "Raw SQL only. Markdown fences or chat preambles fail.",
    },
)

add(
    id="syn-fmt-006",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    statement="SELECT\n    user_id,\n    full_name\nFROM dbo.Users\nWHERE email LIKE N'%@example.com'",
    line_prefix="WHERE email LIKE N'%@example.com'",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "complete-predicate", "empty-policy"],
        "expect_empty": True,
        "must_not_contain": ["AND", "ORDER BY", ";"],
        "rubric": "The WHERE predicate is already complete. Do not chain extra AND/ORDER BY unless clearly needed as the next single unit — with no suffix, empty is safer than inventing filters.",
    },
)

add(
    id="syn-fmt-007",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- with customers in london as a cte, list their orders",
    gold="""WITH london_customers AS (
    SELECT
        CustomerID,
        CompanyName
    FROM dbo.Customers
    WHERE City = N'London'
)
SELECT
    lc.CustomerID,
    lc.CompanyName,
    o.OrderID,
    o.OrderDate,
    o.Freight,
    o.ShipCity
FROM london_customers AS lc
INNER JOIN dbo.Orders AS o
    ON o.CustomerID = lc.CustomerID
ORDER BY
    lc.CompanyName,
    o.OrderDate;""",
    eval={
        "difficulty": "hard",
        "tags": ["cte", "join"],
        "must_contain": ["WITH", "dbo.Customers", "dbo.Orders", "London"],
        "required_objects": ["dbo.Customers", "dbo.Orders"],
        "rubric": "Intent allows CTEs. Do not repeat a CTE name already in recent prefix (none here).",
    },
)

add(
    id="syn-fmt-008",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- meals ranked by protein within category",
    gold="""SELECT
    meal_id,
    meal_name,
    category,
    protein_g,
    calories,
    ROW_NUMBER() OVER (
        PARTITION BY category
        ORDER BY protein_g DESC
    ) AS protein_rank
FROM dbo.Meals;""",
    eval={
        "difficulty": "medium",
        "tags": ["window-function"],
        "must_contain": ["ROW_NUMBER", "PARTITION BY", "dbo.Meals"],
        "required_objects": ["dbo.Meals"],
        "rubric": "Window rank within category using listed protein_g.",
    },
)

add(
    id="syn-fmt-009",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- delete failed payments older than 90 days",
    gold="""DELETE
FROM dbo.Payments
WHERE payment_status IN (N'failed', N'declined')
  AND payment_date < DATEADD(DAY, -90, GETDATE());""",
    eval={
        "difficulty": "medium",
        "tags": ["dml", "delete"],
        "must_contain": ["DELETE", "dbo.Payments"],
        "must_not_contain": ["TRUNCATE", "dbo.Users"],
        "required_objects": ["dbo.Payments"],
        "rubric": "DELETE from Payments with listed filters. Do not delete Users.",
    },
)

add(
    id="syn-fmt-010",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- insert a high protein chicken bowl meal",
    gold="""INSERT INTO dbo.Meals (
    meal_id,
    meal_name,
    category,
    calories,
    protein_g,
    carbs_g,
    fats_g,
    dietary_preference
)
VALUES (
    NEWID(),
    N'Chicken bowl',
    N'entree',
    520,
    42.00,
    38.00,
    18.00,
    N'omnivore'
);""",
    eval={
        "difficulty": "medium",
        "tags": ["dml", "insert"],
        "must_contain": ["INSERT", "dbo.Meals", "NEWID()"],
        "must_not_contain": ["password_hash"],
        "required_objects": ["dbo.Meals"],
        "rubric": "INSERT only listed Meals columns. meal_id is uniqueidentifier — use NEWID(), do not invent IDENTITY.",
    },
)
