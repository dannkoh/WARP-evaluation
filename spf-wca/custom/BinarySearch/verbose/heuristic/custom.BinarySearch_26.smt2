(declare-const t Int)
(declare-const in22 Int)
(declare-const in24 Int)
(declare-const in12 Int)
(declare-const in25 Int)
(declare-const in19 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in12 t))  ( <  in12 t)) (not ( = in19 t)))  ( <  in19 t)) (not ( = in22 t)))  ( <  in22 t)) (not ( = in24 t)))  ( <  in24 t)) (not ( = in25 t)))  ( <  in25 t)))

(check-sat)
(get-model)