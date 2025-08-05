(declare-const t Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and  ( >=  in0 -10)  ( <=  in0 10))  ( >=  in1 -10))  ( <=  in1 10))  ( >=  in2 -10))  ( <=  in2 10))  ( >=  in3 -10))  ( <=  in3 10))  ( >=  in4 -10))  ( <=  in4 10))  ( >=  t -100))  ( <=  t 100)) (not ( = in0 t))) (not ( = ( +  in1 in0) t))) (not ( = ( +  in2 ( +  in1 in0)) t))) (not ( = ( +  in3 ( +  in2 ( +  in1 in0))) t))) (not ( = ( +  in4 ( +  in3 ( +  in2 ( +  in1 in0)))) t))) (not ( = in1 t))) (not ( = ( +  in2 in1) t))) (not ( = ( +  in3 ( +  in2 in1)) t))) (not ( = ( +  in4 ( +  in3 ( +  in2 in1))) t))) (not ( = in2 t))) (not ( = ( +  in3 in2) t))) (not ( = ( +  in4 ( +  in3 in2)) t))) (not ( = in3 t))) (not ( = ( +  in4 in3) t))) (not ( = in4 t))))

(check-sat)
(get-model)