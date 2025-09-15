(declare-const in Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)

(assert (and (and (and (and (and (and (and (and (and (and (and  ( >=  in0 in1)  ( >  in0 in1))  ( >=  in0 in2))  ( >  in0 in2))  ( >=  in1 in2))  ( >  in1 in2)) (not ( = in0 in)))  ( >=  in0 in)) (not ( = in1 in)))  ( >=  in1 in)) (not ( = in2 in)))  ( >=  in2 in)))

(check-sat)
(get-model)