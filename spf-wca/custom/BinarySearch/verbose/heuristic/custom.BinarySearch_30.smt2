(declare-const t Int)
(declare-const in22 Int)
(declare-const in26 Int)
(declare-const in14 Int)
(declare-const in28 Int)
(declare-const in29 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in14 t))  ( <  in14 t)) (not ( = in22 t)))  ( <  in22 t)) (not ( = in26 t)))  ( <  in26 t)) (not ( = in28 t)))  ( <  in28 t)) (not ( = in29 t)))  ( <  in29 t)))

(check-sat)
(get-model)