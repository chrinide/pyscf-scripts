#! /bin/csh
#
rm -f fichero neq k p ee eec eexc own net int eadd eeff summary
echo $1 > fichero
awk '/# Point Group/ {isim=$4; printf "%s\n",isim;exit}' $1 >> neq
awk '/Number of Inequivalent atoms/ {printf "%s\n",$6}' $1 >> neq
awk '/Number of Centers/ {ncent=$5; printf "%s\n",ncent;exit}' $1 >> neq
set ncent = ` awk '/Number of Centers/ {print $5}' $1 `
set i = 1
while ($i <= $ncent)
    set rec0 = `awk '/Cartesian coordinates of centers/{print NR;exit}' $1`
    @ rec0 = $rec0 + 1 + $i
    awk 'BEGIN {r0="'$rec0'"} NR==r0 \\
       {printf "%14.8f %14.8f %14.8f %s\n",$3,$4,$5,$2}' $1 >> neq
  @ i++
end
awk '/kinetic energy          =/ {printf "%14.6f\n",$4}' < $1 >> k
awk '/potential energy        =/ {printf "%14.6f\n",$4}' < $1 >> p
awk '/electron repulsion      =/ {printf "%14.6f\n",$4}' < $1 >> ee
awk '/---coulomb              =/ {printf "%14.6f\n",$3}' < $1 >> eec
awk         '/corr            =/ {printf "%14.6f\n",$3}' < $1 >> eexc
awk '/el-own-nuc attraction   =/ {printf "%14.6f\n",$4}' < $1 >> own
awk '/net energy              =/ {printf "%14.6f\n",$4}' < $1 >> net
awk '/interaction energy      =/ {printf "%14.6f\n",$4}' < $1 >> int
awk '/additive energy         =/ {printf "%14.6f\n",$4}' < $1 >> eadd
awk '/effective energy        =/ {printf "%14.6f\n",$4}' < $1 >> eeff
cat neq k p ee eec eexc own net int eadd eeff > mono
rm neq k p ee eec eexc own net int eadd eeff 
rm -f int 
awk '/Interaction with atom/ {getline;printf "%14.6f %14.6f %14.6f %14.6f %14.6f\n",$2,$3,$4,$5,$6 }' < $1 >> int
awk '/Interaction with atom/ {getline;getline;printf "%14.6f %14.6f\n",$5,$6 }' < $1 >> int

set i = 1
while ($i <= $ncent)
    set rec0 = `awk '/Atom      Charge      Eadd/{print NR;exit}' $1`
    @ rec0 = $rec0 + 1 + $i
    awk 'BEGIN {r0="'$rec0'"} NR==r0 \\
       {printf "%14.6f %14.6f %14.6f %14.6f %14.6f %14.6f\n",$2,$3,$4,$5,$6,$7}' $1 >> summary
  @ i++
end
cat fichero mono int summary > data-from-promolden
rm -f fichero mono int summary 
