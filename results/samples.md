# Qualitative Samples (10 Dev Examples)

## Correct Predictions (5 Examples)

### Example 1
- **Question:** Who was the pole position for the rnd equalling 12?
- **Gold SQL:** `SELECT "Pole position" FROM table WHERE "Rnd" = '12'`
- **Predicted SQL:** `SELECT "Pole position" FROM table WHERE "Rnd" = '12'`

### Example 2
- **Question:** What was the score of the game on November 12?
- **Gold SQL:** `SELECT "Score" FROM table WHERE "Date" = 'November 12'`
- **Predicted SQL:** `SELECT "Score" FROM table WHERE "Date" = 'November 12'`

### Example 3
- **Question:** What was the report of the Belgian Grand Prix?
- **Gold SQL:** `SELECT "Report" FROM table WHERE "Grand Prix" = 'Belgian Grand Prix'`
- **Predicted SQL:** `SELECT "Report" FROM table WHERE "Grand Prix" = 'Belgian Grand Prix'`

### Example 4
- **Question:** When was the successor seated when the district was California 10th?
- **Gold SQL:** `SELECT "Date successor seated" FROM table WHERE "District" = 'California 10th'`
- **Predicted SQL:** `SELECT "Date successor seated" FROM table WHERE "District" = 'California 10th'`

### Example 5
- **Question:** What was the lowest # of total votes?
- **Gold SQL:** `SELECT MIN("# of total votes") FROM table`
- **Predicted SQL:** `SELECT MIN("# of total votes") FROM table`

---

## Failure Cases (5 Examples)

### Failure Example 1
- **Question:** What position does the player who played for butler cc (ks) play?
- **Gold SQL:** `SELECT "Position" FROM table WHERE "School/Club Team" = 'Butler CC (KS)'`
- **Predicted SQL:** `SELECT "No." FROM table WHERE "Player" = 'butler cc'`
- **Failure Mode:** Schema Column Index Misalignment (Wrong SELECT & WHERE columns)
- **Diagnosis:** The model selects `<c1>` instead of `<c3>` for the target column and pairs condition string `'butler cc'` with `<c0>` (`Player`) rather than `<c5>` (`School/Club Team`), reflecting cross-attention ambiguity across table headers.

### Failure Example 2
- **Question:** How many schools did player number 3 play at?
- **Gold SQL:** `SELECT COUNT("School/Club Team") FROM table WHERE "No." = '3'`
- **Predicted SQL:** `SELECT COUNT("Nationality") FROM table WHERE "Position" = '3'`
- **Failure Mode:** Wrong Column Selection (SELECT and WHERE)
- **Diagnosis:** The aggregation operator `COUNT` is correctly predicted (`agg: 3`), but both the target column (`<c2>` vs `<c5>`) and the condition column (`<c3>` vs `<c1>`) suffer from schema-grounding shifts.

### Failure Example 3
- **Question:** What school did player number 21 play for?
- **Gold SQL:** `SELECT "School/Club Team" FROM table WHERE "No." = '21'`
- **Predicted SQL:** `SELECT "Player" FROM table WHERE "Position" = '21'`
- **Failure Mode:** Wrong Target and Condition Columns
- **Diagnosis:** The model correctly identifies the entity literal `'21'` and operator `=`, but fails to link the question token "school" to `<c5>` and "number" to `<c1>`.

### Failure Example 4
- **Question:** Who is the player that wears number 42?
- **Gold SQL:** `SELECT "Player" FROM table WHERE "No." = '42'`
- **Predicted SQL:** `SELECT "Player" FROM table WHERE "Position" = '42'`
- **Failure Mode:** Wrong Condition Column Index
- **Diagnosis:** The SELECT column (`"Player"`, `<c0>`), condition operator (`=`), and numeric literal (`'42'`) match the gold query, but the condition column index is misclassified as `<c3>` (`Position`) instead of `<c1>` (`No.`).

### Failure Example 5
- **Question:** What player played guard for toronto in 1996-97?
- **Gold SQL:** `SELECT "Player" FROM table WHERE "Position" = 'Guard' AND "Years in Toronto" = '1996-97'`
- **Predicted SQL:** `SELECT "Player" FROM table WHERE "Position" = '1996-97'`
- **Failure Mode:** Multi-Condition Dropping & Schema Value Collision
- **Diagnosis:** The model only generates a single WHERE clause, omitting `"Position" = 'Guard'`, and places the temporal literal `'1996-97'` into the `"Position"` column instead of `"Years in Toronto"`.