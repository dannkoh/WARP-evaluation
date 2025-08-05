(declare-const in8 Int)
(declare-const t Int)
(declare-const in12 Int)
(declare-const in15 Int)
(declare-const in14 Int)
(declare-const in16 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in8 t))  ( <  in8 t)) (not ( = in12 t)))  ( <  in12 t)) (not ( = in14 t)))  ( <  in14 t)) (not ( = in15 t)))  ( <  in15 t)) (not ( = in16 t)))  ( <  in16 t)))

(check-sat)
(get-model)