(declare-const in20 Int)
(declare-const t Int)
(declare-const in21 Int)
(declare-const in10 Int)
(declare-const in16 Int)
(declare-const in19 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in10 t))  ( <  in10 t)) (not ( = in16 t)))  ( <  in16 t)) (not ( = in19 t)))  ( <  in19 t)) (not ( = in20 t)))  ( <  in20 t)) (not ( = in21 t)))  ( <  in21 t)))

(check-sat)
(get-model)