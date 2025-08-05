(declare-const in Int)
(declare-const in0 Int)

(assert  ( >  in in0))

(check-sat)
(get-model)