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
!(addToSpace ((A FALSE) (B TRUE) (C TRUE)))
;!(collapse (match &self ($val) $val))


