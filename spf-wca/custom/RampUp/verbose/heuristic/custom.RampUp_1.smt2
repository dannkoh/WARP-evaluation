(declare-const in0 Int)

(assert  ( <  in0 0))

(check-sat)
(get-model)