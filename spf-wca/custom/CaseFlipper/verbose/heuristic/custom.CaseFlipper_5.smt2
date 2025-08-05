(declare-const ch0 Int)
(declare-const ch2 Int)
(declare-const ch1 Int)
(declare-const ch4 Int)
(declare-const ch3 Int)

(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and  ( <  ch0 97)  ( >=  ch0 65))  ( <=  ch0 90))  ( <  ch1 97))  ( >=  ch1 65))  ( <=  ch1 90))  ( <  ch2 97))  ( >=  ch2 65))  ( <=  ch2 90))  ( <  ch3 97))  ( >=  ch3 65))  ( <=  ch3 90))  ( <  ch4 97))  ( >=  ch4 65))  ( <=  ch4 90)))

(check-sat)
(get-model)