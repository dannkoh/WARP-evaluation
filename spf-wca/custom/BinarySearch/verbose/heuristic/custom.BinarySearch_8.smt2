(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in7 Int)
(declare-const t Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (and (and (not ( = in3 t))  ( <  in3 t)) (not ( = in5 t)))  ( <  in5 t)) (not ( = in6 t)))  ( <  in6 t)) (not ( = in7 t)))  ( <  in7 t)))

(check-sat)
(get-model)