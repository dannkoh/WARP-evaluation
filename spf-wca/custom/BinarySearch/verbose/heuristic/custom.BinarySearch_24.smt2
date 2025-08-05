(declare-const in20 Int)
(declare-const t Int)
(declare-const in22 Int)
(declare-const in11 Int)
(declare-const in23 Int)
(declare-const in17 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in11 t))  ( <  in11 t)) (not ( = in17 t)))  ( <  in17 t)) (not ( = in20 t)))  ( <  in20 t)) (not ( = in22 t)))  ( <  in22 t)) (not ( = in23 t)))  ( <  in23 t)))

(check-sat)
(get-model)