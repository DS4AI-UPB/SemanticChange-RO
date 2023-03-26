from representations import sgns_op, sgns_wi, elmo_with_precomp
from sys import argv

model = None
if argv[1] == "sgns_op":
    model = sgns_op.SGNS()
elif argv[1] == "sgns_wi":
    model = sgns_wi.SGNS()
elif argv[1] == "elmo_with_precomp":
    model = elmo_with_precomp.ELMo()
else:
    print("Model not supported")
    exit(1)
model.do_test(argv[2])