"""One-function, offline PDB/PE inspection; no compilation or model execution."""
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import uuid

ROOT=Path(__file__).resolve().parents[2]
BUILD=ROOT/'artifacts/diagnostics/attention64-v2/benchmark-build'
PDB=Path('D:/falcon-ocr-rust-builds/attention64-v2/release/examples/ocr_bench.pdb')
PDBUTIL=Path('C:/Program Files/LLVM/bin/llvm-pdbutil.exe')
OBJDUMP=Path('C:/Program Files/LLVM/bin/llvm-objdump.exe')
OUT=BUILD/'compiled-head'
NAME='falcon_ocr::kernels::attention64_candidate::head'
BUILD_SHA='cde63eeebb2eebad095425a5b7a1c3a94eef7b20ac7bef540366b84e4f030d6c'

def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(p.read_bytes())
def need(ok,s):
    if not ok:raise ValueError(s)
def save(name,text):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def command(args):return subprocess.check_output([str(x) for x in args],text=True)

rawbuild=(BUILD/'build.json').read_bytes();need(digest(rawbuild)==BUILD_SHA,'Build changed')
build=json.loads(rawbuild);exe=BUILD/build['binary'];raw=exe.read_bytes()
need(digest(raw)==build['binary_sha256'],'Binary changed')
bound={str(BUILD/'build.json'):BUILD_SHA,str(exe):digest(raw),str(PDB):sha(PDB),
       str(PDBUTIL):sha(PDBUTIL),str(OBJDUMP):sha(OBJDUMP),str(Path(__file__)):sha(Path(__file__))}
OUT.mkdir(exist_ok=False)
preserved=BUILD/'ocr_bench.pdb'
with preserved.open('xb') as f:f.write(PDB.read_bytes())
need(sha(preserved)==bound[str(PDB)],'PDB copy changed')
summary=command([PDBUTIL,'dump','-summary',preserved]);save('pdb-summary.txt',summary)
symbols=command([PDBUTIL,'dump','-symbols',preserved])
matches=list(re.finditer(r'S_LPROC32[^\n]*`'+re.escape(NAME)+r'`\n([^\n]+)',symbols))
need(len(matches)==1,'Function symbol absent/ambiguous')
symbol=matches[0].group();save('head-symbol.txt',symbol+'\n')
address=re.search(r'addr = (\d+):(\d+), code size = (\d+)',symbol)
need(address is not None,'No function extent')
section,offset,size=map(int,address.groups())
# Parse PE debug directory; do not infer PDB identity from a filename alone.
pe=struct.unpack_from('<I',raw,0x3c)[0];need(raw[pe:pe+4]==b'PE\0\0','Invalid PE')
count=struct.unpack_from('<H',raw,pe+6)[0];opt_size=struct.unpack_from('<H',raw,pe+20)[0]
opt=pe+24;need(struct.unpack_from('<H',raw,opt)[0]==0x20b,'Expected PE32+')
base=struct.unpack_from('<Q',raw,opt+24)[0]
sections=[]
for i in range(count):
    pos=opt+opt_size+40*i
    virtual_size,rva,rawsize,fileoffset=struct.unpack_from('<IIII',raw,pos+8)
    sections.append((rva,rawsize,fileoffset,raw[pos:pos+8].rstrip(b'\0').decode()))
def file_offset(rva):
    found=[off+rva-start for start,size,off,_ in sections if start<=rva<start+size]
    need(len(found)==1,'RVA unmapped/ambiguous');return found[0]
debug_rva,debug_size=struct.unpack_from('<II',raw,opt+112+6*8)
debug_offset=file_offset(debug_rva);rsds=[]
for pos in range(debug_offset,debug_offset+debug_size,28):
    kind,length,_,pointer=struct.unpack_from('<IIII',raw,pos+12)
    if kind==2 and raw[pointer:pointer+4]==b'RSDS':
        rsds.append({'guid':str(uuid.UUID(bytes_le=raw[pointer+4:pointer+20])).upper(),
                     'age':struct.unpack_from('<I',raw,pointer+20)[0],
                     'original_path':raw[pointer+24:pointer+length].rstrip(b'\0').decode()})
need(len(rsds)==1,'Expected one CodeView RSDS record')
identity=rsds[0]
need('GUID: {'+identity['guid']+'}' in summary and re.search(r'Age:\s*'+str(identity['age'])+r'\b',summary),'PE/PDB identity mismatch')
need(1<=section<=len(sections),'Wrong symbol section')
start=base+sections[section-1][0]+offset;stop=start+size
args=[OBJDUMP,'-d','--no-show-raw-insn',f'--start-address={hex(start)}',f'--stop-address={hex(stop)}',exe]
disassembly=command(args);save('head-disassembly.txt',disassembly)
instructions=[]
for line in disassembly.splitlines():
    m=re.match(r'^([0-9a-f]+):\s+(\w+)\s*(.*)',line)
    if m:instructions.append((int(m[1],16),m[2],m[3]))
need(instructions and instructions[0][0]==start and instructions[-1][0]<stop,'Unexpected disassembly extent')
calls=[{'address':hex(a),'operand':op} for a,m,op in instructions if m.startswith('call')]
indirect=[c for c in calls if c['operand'].lstrip().startswith('*')]
dot=[(a,m,op) for a,m,op in instructions if 0x1400b16c1<=a<=0x1400b176d]
axpy=[(a,m,op) for a,m,op in instructions if 0x1400b1929<=a<=0x1400b19cd]
need(sum(m.startswith('vfmadd') for _,m,_ in dot)==8,'Unexpected dot FMA inventory')
need(sum(m.startswith('vfmadd') for _,m,_ in axpy)==8,'Unexpected AXPY FMA inventory')
need(not indirect,'Head retains indirect calls')
publics=command([PDBUTIL,'dump','-publics',preserved])
public_evidence=[]
for name in ['expf','logf','memset']:
    m=re.search(r'[^\n]*S_PUB32[^\n]*`'+name+r'`\n[^\n]+',publics)
    need(m is not None,'Missing direct-call symbol '+name);public_evidence.append(m.group())
save('direct-call-symbols.txt','\n'.join(public_evidence)+'\n')
need(all(sha(Path(p))==h for p,h in bound.items()),'Input/tool/source closure changed')
need(sha(preserved)==bound[str(PDB)],'Preserved PDB changed')
report={'kind':'attention64-compiled-head-v1','status':'fixed_width_inline_confirmed',
    'input_sha256':bound,'preserved_pdb':str(preserved.relative_to(ROOT)),'pdb_sha256':sha(preserved),
    'pe_codeview':identity,'pdb_identity_matches_pe':True,'function':NAME,'section':section,
    'section_offset_decimal':offset,'code_bytes':size,'virtual_address':hex(start),'stop_address':hex(stop),
    'objdump_command':[str(x) for x in args],'dot_fma_instructions':8,'axpy_fma_instructions':8,
    'indirect_calls':indirect,'calls':calls,'source_and_input_closure':True,
    'observations':['Dot is unrolled into eight YMM FMAs: four zero accumulations and four second products.',
        'Three YMM adds then half extraction/add and two horizontal adds retain the original dot tree.',
        'AXPY is unrolled into eight YMM FMA/store blocks in offsets0,32,...224 bytes.',
        'No indirect call exists in the head. Dot and AXPY are inline; scalar exp/log calls remain direct.',
        'This checks one compiled function; no new build, inference, profile or benchmark was run.'],
    'limitations':['Objdump COFF nearest-export labels name an unrelated onig export; PDB section/offset/extent identifies this head.',
        'Source and operator/smoke checks remain separate evidence; this is not a whole-binary equivalence proof.',
        'No default promotion or additional performance claim.'],
    'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
save('symbol-receipt.json',json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'pdb_sha256':sha(preserved),'receipt_sha256':sha(OUT/'symbol-receipt.json')}))
