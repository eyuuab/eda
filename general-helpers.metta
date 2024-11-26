
(= (addToSpace $row $space)
    (if (== $row ())
        ()
        (let*
            (
             (() (println! (AddToSpace Params: $row)))
            ;  (() (println! (Space content: (collapse (match $space $x $x)))))

             ($head (car-atom $row))

             ($tail (addToSpace (cdr-atom $row) $space)) 
             (() (println! (Adding: $head)))
             (() (println! (Next Param: (cdr-atom $row))))
             (() (println! ""))   
            ;  (() (println! ($head)))      
            )
            (add-atom $space ($head))
        )
    )
)

(= (addToSpace1 $row $space) (add-atom $space $row))

(= (removeFromSpace1 $row $space) (remove-atom $space $row))

(= (sum $tuple)
    (if (== $tuple ())
        0
        (+ (car-atom $tuple) (sum (cdr-atom $tuple)))
    )
)

; replace expersion with Truth value from the space
(= (replaceWithTruth $expr $space)
    (if (== (get-metatype $expr) Expression) 
        (let* (
            ($op (car-atom $expr))
            ($tuple (cdr-atom $expr)) 
            (() (println! (ReplaceWithTruth Params: $expr)))
            (() (println! (ReplacewithTruth Space: (collapse (match $space $x $x)))))
            (() (println! (Tuple: $tuple)))
             ($substituted-args (collapse (replaceWithTruth (superpose $tuple) $space)) )
             (() (println! (SubstitutedArgs: $substituted-args))) 
            (() (println! ""))

            ($tupleWithOp (cons-atom $op $substituted-args))
        )
            $tupleWithOp
        )
         (unify $space ($expr $value) $value 'undefined') 
    )
)

;a function to evaluate the boolean expression
(= (eval $expr)
    (if (== (get-metatype $expr) Grounded)
        $expr
        (let* (
            ($op (car-atom $expr))
            ($tuple (cdr-atom $expr))
        )
            (if (== $op AND)
                (evalAnd $tuple)
                (if (== $op OR)
                    (evalOr $tuple)
                    ERROR
                )
            )
        )
    )
)

(= (evalOr $args)
    (if (== $args ())
        False
        (let* (
            ($first (car-atom $args))
            ($rest (cdr-atom $args))
        )
            (if (== (eval $first) True)
                True
                (evalOr $rest)
            )
        )
    )
)

(= (evalAnd $args)
    (if (== $args ())
        True
        (let* (
            ($first (car-atom $args))
            ($rest (cdr-atom $args))
        )
            (if (== (eval $first) False)
                False
                (evalAnd $rest)
            )
        )
    )
)

(= (removeFromSpace $inputs $space)
    (if (== $inputs ())
        ()
        (let* (
            ($currentInput (car-atom $inputs)) 
            ($remaining (removeFromSpace (cdr-atom $inputs) $space))
        )
        (remove-atom $space ($currentInput))
        )
    )
)

; Function to count operators in a boolean expression
(= (countOperators $expr)
    (if (== (get-metatype $expr) Grounded) ; Base case: no operator in a grounded expression
        0
        (let* (
            ($op (car-atom $expr)) ; Extract the operator
            ($args (cdr-atom $expr)) ; Extract the arguments
            ($nestedCount (countInList $args)) ; Recursively count operators in arguments
        )
            (+ 1 $nestedCount) ; Count the current operator and add nested counts
        )
    )
)

; Helper function to count operators in a list of expressions
(= (countInList $args)
    (if (== $args ())
        0
        (let* (
            ($first (car-atom $args))
            ($rest (cdr-atom $args))
        )
            (+ (countOperators $first) (countInList $rest)) 
            2; Count in head and tail
        )
    )
)



(= (inputGenerator $var)
    (if (== $var ())
        ()
        (let* (
            ($head (car-atom $var))
            ($tail (inputGenerator (cdr-atom $var)))
        )
        
        )
    )
)

(= (test $tuple)
    (if (== $tuple ())
        ()
        (let* (
            ($head (car-atom $tuple))
            ($tail (test (cdr-atom $tuple)))
        )
            (addToSpace $head &self)
        )
    )
)

 ;!(test ((1 2 3 4) (A B C D) (a b c d)))

; ; !(eval (AND True False True))  ; False
; ; !(eval (OR False False True))  ; True
; ; !(eval (AND (OR True False) True (AND False True)))  ; False
; ; !(eval (AND True True True))  ; True
; ; !(eval (OR (AND False True) (OR True False)))  ; True
; ; !(countOperators (AND A (OR B C) (AND D E)))

; ; !(addToSpace (1 2 3 4) &self)
; !(collapse (match &self ($val) $val))
; !(removeFromSpace (1 2 3 4) &self)
; !(collapse (match &self ($val) $val))
; !(replaceWithTruth (AND A (OR B C)) &self)

;!(complexity (AND True (OR False True)))  ; 3
