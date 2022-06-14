#/usr/bin

if [ $# -ge 1 ]; then 

    if [ $1 = 'train' ]; then 
        
        if [ $# -eq 2 ]; then
            id=$2
            python train.py --id $id
        else
            python train.py --id 'lab'
        fi
    
    elif [ $1 = 'val' ]; then

        if [ $# -eq 2 ]; then
            id=$2
            python val.py --id $id
        else
            python val.py --id 'lab'
        fi
    
    else 

        echo "unknown function : $1"
        
    fi
else 

    echo 'must be at least one arg'

fi