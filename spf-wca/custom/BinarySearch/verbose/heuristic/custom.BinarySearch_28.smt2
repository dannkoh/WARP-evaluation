(declare-const in20 Int)
(declare-const t Int)
(declare-const in24 Int)
(declare-const in13 Int)
(declare-const in26 Int)
(declare-const in27 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in13 t))  ( <  in13 t)) (not ( = in20 t)))  ( <  in20 t)) (not ( = in24 t)))  ( <  in24 t)) (not ( = in26 t)))  ( <  in26 t)) (not ( = in27 t)))  ( <  in27 t)))

(check-sat)
(get-model)