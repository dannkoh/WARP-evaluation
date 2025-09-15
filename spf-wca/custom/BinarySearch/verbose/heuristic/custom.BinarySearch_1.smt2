(declare-const t Int)
(declare-const in0 Int)

(assert (and (not ( = in0 t))  ( <  in0 t)))

(check-sat)
(get-model)