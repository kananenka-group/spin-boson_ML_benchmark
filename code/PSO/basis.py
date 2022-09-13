
def save(basis, filename):
   f = open(filename,"w")
   N = len(basis)
   f.write(" %d \n"%(N))
   for bas in basis:
     Nb = len(bas)
     f.write(" %d \n"%(Nb))
     for x in bas:
        f.write(" %s   %15.9f \n"%(x[0],x[1]))
   f.close()
   return

def read(filename):

   f = open(filename, "r")
   line = f.readline()
   line2 = line.split()
   N = int(line2[0])

   basis = []

   for nx in range(N):
      line = f.readline()
      line2 = line.split()
      nb = int(line2[0])
      bas = []
      for ny in range(nb):
         line = f.readline()
         line2 = line.split()
         sym = line2[0] 
         val = float(line2[1])
         exp = []
         exp.append(sym)
         exp.append(val)
         bas.append(exp)
      basis.append(bas)

   f.close()
   return basis

def write_basis(bas, gen, idx):
   filename = "./basis/bas_gen" + str(gen) + "_id" + str(idx) + ".dat"
   f = open(filename, "w")
   for p in range(len(bas)):
      v = bas[p]
      f.write("    %18.9f \n"%(v))
   f.close()
   return

def read_start_basis(start_file):
   f = open(start_file,"r")
   basis = []
   for line in f:
      ls = []
      line2 = line.split()
      sh_ = line2[0]
      ex_ = float(line2[1])
      ls.append(sh_)
      ls.append(ex_)
      basis.append(ls)
   f.close() 

   return basis

def read_pseudopotential(pp_file):
   """
      Since the pseudopotential part
      will be left untouched will simply
      read it here

   """
   f = open(pp_file,"r")
   pp = f.read()
   f.close()
   return pp 

def read_atom_basis(basis_file):
   f = open(basis_file,"r")
   bs = f.read()
   f.close()
   return bs
