(declare-const in20 Int)
(declare-const t Int)
(declare-const in13 Int)
(declare-const in23 Int)
(declare-const in26 Int)
(declare-const in25 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in13 t))  ( <  in13 t)) (not ( = in20 t)))  ( <  in20 t)) (not ( = in23 t)))  ( <  in23 t)) (not ( = in25 t)))  ( <  in25 t)) (not ( = in26 t)))  ( <  in26 t)))

(check-sat)
(get-model)