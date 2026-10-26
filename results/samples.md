# Qualitative samples (dev set, greedy decoding)

## Correct
- **Q:** How many schools did player number 3 play at?
  - Gold: `SELECT COUNT(School/Club Team) FROM table WHERE No. = '3'`
  - Pred: `SELECT COUNT(School/Club Team) FROM table WHERE No. = '3'`
- **Q:** If the average is 50.16, who is the player?
  - Gold: `SELECT Player FROM table WHERE Average = '50.16'`
  - Pred: `SELECT Player FROM table WHERE Average = '50.16'`
- **Q:** What is the average silver for golds over 2, ranks of 1, and bronzes over 0?
  - Gold: `SELECT AVG(Silver) FROM table WHERE Gold > '2' AND Rank = '1' AND Bronze > '0'`
  - Pred: `SELECT AVG(Silver) FROM table WHERE Gold > '2' AND Rank = '1' AND Bronze > '0'`
- **Q:** What is the date where the round is R3?
  - Gold: `SELECT Date FROM table WHERE Round = 'r3'`
  - Pred: `SELECT Date FROM table WHERE Round = 'r3'`
- **Q:** Where was the game played on 20 may?
  - Gold: `SELECT Venue FROM table WHERE Date = '20 may'`
  - Pred: `SELECT Venue FROM table WHERE Date = '20 may'`

## Wrong
- **Q:** What position does the player who played for butler cc (ks) play?
  - Gold: `SELECT Position FROM table WHERE School/Club Team = 'Butler CC (KS)'`
  - Pred: `SELECT Position FROM table WHERE Player = 'butler cc (ks)'`
  - Failure: wrong WHERE column. The question names the school/club team, but the model put the school name against the Player column.
- **Q:** When did Jacques Chirac stop being a G8 leader?
  - Gold: `SELECT Ended time as senior G8 leader FROM table WHERE Person = 'Jacques Chirac'`
  - Pred: `SELECT Began time as senior G8 leader FROM table WHERE Entered office as Head of State or Government = 'jacques chirac stop'`
  - Failure: wrong SELECT column, wrong WHERE column. The model copied the extra word 'stop' into the value, picked the wrong column for the condition, and chose 'Began time' where the question asks when the leader stopped.
- **Q:** What official or native languages are spoken in the country whose capital city is Canberra?
  - Gold: `SELECT Official or native language(s) (alphabet/script) FROM table WHERE Capital ( exonym ) = 'Canberra'`
  - Pred: `SELECT Capital ( endonym ) FROM table WHERE Capital ( exonym ) = 'canberra'`
  - Failure: wrong SELECT column. The question asks for languages, but the model selected the capital's local-name column. The table has several similar capital and country columns.
- **Q:** What is the amount of trees, that require replacement when prevailing types, % is pine — 29.37 poplar — 26.12 acer negundo — 13.2?
  - Gold: `SELECT Amount of trees, that require replacement FROM table WHERE Prevailing types, % = 'Pine — 29.37 Poplar — 26.12 Acer negundo — 13.2'`
  - Pred: `SELECT COUNT(Amount of old trees) FROM table WHERE Total amount of trees = 'pine — 29.37 poplar — 26.12'`
  - Failure: wrong SELECT column, wrong aggregate, wrong WHERE column. A very long question with a long value. The model got the selected column, the aggregate and the condition column wrong, and cut the value short.
- **Q:** How many foreigners in percentage terms had a population of 4.911?
  - Gold: `SELECT COUNT(Foreign nationals in %) FROM table WHERE Population = '4.911'`
  - Pred: `SELECT Foreign nationals in % FROM table WHERE Population = '4.911'`
  - Failure: wrong aggregate. The question starts with 'How many', so the answer needed COUNT, but the model returned the column value with no aggregate.