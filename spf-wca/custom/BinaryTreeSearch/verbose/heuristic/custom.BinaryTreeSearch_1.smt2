(declare-const in Int)
(declare-const in0 Int)

(assert (and (not ( = in0 in))  ( >=  in0 in)))

(check-sat)
(get-model)