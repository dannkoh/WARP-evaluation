(declare-const ch0 Int)
(declare-const ch2 Int)
(declare-const ch1 Int)

(assert (and (and (and (and (and (and (and (and  ( <  ch0 97)  ( >=  ch0 65))  ( <=  ch0 90))  ( <  ch1 97))  ( >=  ch1 65))  ( <=  ch1 90))  ( <  ch2 97))  ( >=  ch2 65))  ( <=  ch2 90)))

(check-sat)
(get-model)