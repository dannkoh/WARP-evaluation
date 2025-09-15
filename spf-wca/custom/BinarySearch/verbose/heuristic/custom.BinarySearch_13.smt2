(declare-const in6 Int)
(declare-const t Int)
(declare-const in11 Int)
(declare-const in9 Int)
(declare-const in12 Int)

(assert (and (and (and (and (and (and (and (not ( = in6 t))  ( <  in6 t)) (not ( = in9 t)))  ( <  in9 t)) (not ( = in11 t)))  ( <  in11 t)) (not ( = in12 t)))  ( <  in12 t)))

(check-sat)
(get-model)