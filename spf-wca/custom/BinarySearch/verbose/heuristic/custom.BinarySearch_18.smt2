(declare-const in8 Int)
(declare-const t Int)
(declare-const in13 Int)
(declare-const in15 Int)
(declare-const in17 Int)
(declare-const in16 Int)

(assert (and (and (and (and (and (and (and (and (and (not ( = in8 t))  ( <  in8 t)) (not ( = in13 t)))  ( <  in13 t)) (not ( = in15 t)))  ( <  in15 t)) (not ( = in16 t)))  ( <  in16 t)) (not ( = in17 t)))  ( <  in17 t)))

(check-sat)
(get-model)