(declare-const in6 Int)
(declare-const t Int)
(declare-const in10 Int)
(declare-const in13 Int)
(declare-const in12 Int)

(assert (and (and (and (and (and (and (and (not ( = in6 t))  ( <  in6 t)) (not ( = in10 t)))  ( <  in10 t)) (not ( = in12 t)))  ( <  in12 t)) (not ( = in13 t)))  ( <  in13 t)))

(check-sat)
(get-model)