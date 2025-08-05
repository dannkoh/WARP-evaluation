(declare-const in20 Int)
(declare-const t Int)
(declare-const in10 Int)
(declare-const in15 Int)
(declare-const in19 Int)
(declare-const in18 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in10 t))  ( <  in10 t)) (not ( = in15 t)))  ( <  in15 t)) (not ( = in18 t)))  ( <  in18 t)) (not ( = in19 t)))  ( <  in19 t)) (not ( = in20 t)))  ( <  in20 t)))

(check-sat)
(get-model)