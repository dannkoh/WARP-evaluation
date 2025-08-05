(declare-const in Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and  ( >=  in0 in1)  ( >  in0 in1))  ( >=  in0 in2))  ( >  in0 in2))  ( >=  in1 in2))  ( >  in1 in2))  ( >=  in0 in3))  ( >  in0 in3))  ( >=  in1 in3))  ( >  in1 in3))  ( >=  in2 in3))  ( >  in2 in3)) (not ( = in0 in)))  ( >=  in0 in)) (not ( = in1 in)))  ( >=  in1 in)) (not ( = in2 in)))  ( >=  in2 in)) (not ( = in3 in)))  ( >=  in3 in)))

(check-sat)
(get-model)