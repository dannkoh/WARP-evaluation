(declare-const t Int)
(declare-const in9 Int)
(declare-const in14 Int)
(declare-const in17 Int)
(declare-const in19 Int)
(declare-const in18 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in9 t))  ( <  in9 t)) (not ( = in14 t)))  ( <  in14 t)) (not ( = in17 t)))  ( <  in17 t)) (not ( = in18 t)))  ( <  in18 t)) (not ( = in19 t)))  ( <  in19 t)))

(check-sat)
(get-model)