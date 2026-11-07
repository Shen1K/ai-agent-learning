from pathlib import Path
from urllib.parse import unquote
import re,json,sys
root=Path(sys.argv[1]).resolve()
files=[p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts]
errors=[]; counts={'markdown_links':0,'wiki_links':0,'canvas_file_nodes':0,'canvas_edges':0}
texts={p:p.read_text() for p in files if p.suffix in {'.md','.canvas'}}

def anchors(path):
 text=texts.get(path,'');result=set()
 for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$',text,re.M):
  result.add(heading.casefold())
  plain=re.sub(r'<[^>]+>','',heading).replace('`','')
  slug=re.sub(r'[^\w\-\s\u4e00-\u9fff]','',plain.casefold()).replace(' ','-')
  result.add(slug)
 result.update(a.casefold() for a in re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)',text))
 result.update('^'+a for a in re.findall(r'\^([A-Za-z0-9-]+)\s*$',text,re.M))
 return result

def check(source,target,wiki=False):
 target=unquote(target.strip('<>'))
 if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):return
 base,_,frag=target.partition('#')
 if not base:dest=source
 elif wiki:
  options=[root/base,source.parent/base]
  options += [p.with_suffix('.md') for p in options if not p.suffix]
  matches=[p for p in options if p.is_file()]
  if not matches:matches=[p for p in files if p.name==base or p.stem==base]
  matches=list(dict.fromkeys(p.resolve() for p in matches))
  if not matches: errors.append([str(source.relative_to(root)),target,'missing wiki file']);return
  if len(matches)>1: errors.append([str(source.relative_to(root)),target,'ambiguous wiki file']);return
  dest=matches[0]
 else: dest=(source.parent/base).resolve()
 if not dest.is_file():errors.append([str(source.relative_to(root)),target,'missing local file']);return
 if not dest.is_relative_to(root):errors.append([str(source.relative_to(root)),target,'outside export']);return
 if frag and dest.suffix=='.md' and frag.casefold() not in anchors(dest):
  errors.append([str(source.relative_to(root)),target,'missing anchor'])

for p,text in texts.items():
 if p.suffix=='.canvas':
  data=json.loads(text); ids={n['id'] for n in data['nodes']}
  for n in data['nodes']:
   if n.get('type')=='file':
    counts['canvas_file_nodes']+=1
    check(p,n['file']+n.get('subpath',''),True)
  for edge in data['edges']:
   counts['canvas_edges']+=1
   if edge['fromNode'] not in ids or edge['toNode'] not in ids:errors.append([str(p),'edge','missing endpoint'])
  continue
 text=re.sub(r'^(```|~~~)[^\n]*\n.*?^\1[^\n]*$', '',text,flags=re.S|re.M)
 text=re.sub(r'`[^`\n]+`', '', text)
 for target in re.findall(r'!?\[\[([^\]\n]+)\]\]',text):
  counts['wiki_links']+=1;check(p,target.split('|')[0],True)
 for target in re.findall(r'!?\[[^\]\n]+\]\((<[^>\n]+>|[^)\n]+)\)',text):
  if not re.match(r'<?[a-zA-Z][a-zA-Z0-9+.-]*:',target):
   counts['markdown_links']+=1;check(p,target)
print(json.dumps({'counts':counts,'errors':errors},ensure_ascii=False,indent=2))
sys.exit(bool(errors))
