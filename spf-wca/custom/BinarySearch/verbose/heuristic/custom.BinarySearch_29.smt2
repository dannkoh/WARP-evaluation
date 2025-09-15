(declare-const t Int)
(declare-const in21 Int)
(declare-const in25 Int)
(declare-const in14 Int)
(declare-const in28 Int)
(declare-const in27 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in14 t))  ( <  in14 t)) (not ( = in21 t)))  ( <  in21 t)) (not ( = in25 t)))  ( <  in25 t)) (not ( = in27 t)))  ( <  in27 t)) (not ( = in28 t)))  ( <  in28 t)))

(check-sat)
(get-model)