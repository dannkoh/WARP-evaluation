(declare-const t Int)
(declare-const in21 Int)
(declare-const in24 Int)
(declare-const in23 Int)
(declare-const in12 Int)
(declare-const in18 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in12 t))  ( <  in12 t)) (not ( = in18 t)))  ( <  in18 t)) (not ( = in21 t)))  ( <  in21 t)) (not ( = in23 t)))  ( <  in23 t)) (not ( = in24 t)))  ( <  in24 t)))

(check-sat)
(get-model)