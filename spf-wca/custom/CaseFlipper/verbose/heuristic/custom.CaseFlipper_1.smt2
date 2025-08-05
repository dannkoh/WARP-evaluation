(declare-const ch0 Int)

(assert (and (and  ( <  ch0 97)  ( >=  ch0 65))  ( <=  ch0 90)))

(check-sat)
(get-model)