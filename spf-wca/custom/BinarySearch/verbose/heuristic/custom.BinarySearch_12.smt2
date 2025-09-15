(declare-const in5 Int)
(declare-const in8 Int)
(declare-const in11 Int)
(declare-const t Int)
(declare-const in10 Int)

(assert (and (and (and (and (and (and (and (not ( = in5 t))  ( <  in5 t)) (not ( = in8 t)))  ( <  in8 t)) (not ( = in10 t)))  ( <  in10 t)) (not ( = in11 t)))  ( <  in11 t)))

(check-sat)
(get-model)