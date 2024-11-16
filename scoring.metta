(= (addToSpace $row)
    (if (== $row ())
        ()
        (let*
            (($head (car-atom $row))
             ($tail (addToSpace (cdr-atom $row)))
            )
            (add-atom &self ($head))
        )
    )
)


(= (replaceWithTruth $expr)
    (if (== (get-metatype $expr) Expression) 
        (let* (
            ($op (car-atom $expr))
            ($tuple (cdr-atom $expr)) 
            ($substituted-args (collapse (replaceWithTruth (superpose $tuple)))) 
            ($tupleWithOp (cons-atom $op $substituted-args)) 
        )
            $tupleWithOp
        )
         (unify &self (($expr $value)) $value 'undefined') 
    )
)

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

!(eval (AND True False True))  ; False
!(eval (OR False False True))  ; True
!(eval (AND (OR True False) True (AND False True)))  ; False
!(eval (AND True True True))  ; True
!(eval (OR (AND False True) (OR True False)))  ; True


; !(addToSpace ((A FALSE) (B TRUE) (C TRUE)))
; ;!(collapse (match &self ($val) $val))
; !(replaceWithTruth (AND A (OR B C)))