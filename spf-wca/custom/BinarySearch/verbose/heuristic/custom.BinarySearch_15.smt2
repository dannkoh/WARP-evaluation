(declare-const in7 Int)
(declare-const t Int)
(declare-const in11 Int)
(declare-const in13 Int)
(declare-const in14 Int)

(assert (and (and (and (and (and (and (and (not ( = in7 t))  ( <  in7 t)) (not ( = in11 t)))  ( <  in11 t)) (not ( = in13 t)))  ( <  in13 t)) (not ( = in14 t)))  ( <  in14 t)))

(check-sat)
(get-model)