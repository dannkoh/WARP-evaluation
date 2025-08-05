(declare-const in5 Int)
(declare-const in8 Int)
(declare-const t Int)
(declare-const in10 Int)
(declare-const in9 Int)

(assert (and (and (and (and (and (and (and (not ( = in5 t))  ( <  in5 t)) (not ( = in8 t)))  ( <  in8 t)) (not ( = in9 t)))  ( <  in9 t)) (not ( = in10 t)))  ( <  in10 t)))

(check-sat)
(get-model)