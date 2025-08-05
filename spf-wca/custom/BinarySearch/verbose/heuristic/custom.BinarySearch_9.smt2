(declare-const in6 Int)
(declare-const in8 Int)
(declare-const in7 Int)
(declare-const t Int)
(declare-const in4 Int)

(assert (and (and (and (and (and (and (and (not ( = in4 t))  ( <  in4 t)) (not ( = in6 t)))  ( <  in6 t)) (not ( = in7 t)))  ( <  in7 t)) (not ( = in8 t)))  ( <  in8 t)))

(check-sat)
(get-model)