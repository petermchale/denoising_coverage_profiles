# edit this code from line 93 to print out read and fragment depth, individually
# some resources:
# 1. https://nim-by-example.github.io/
# 2. https://nim-lang.org/docs/tut1.html
# https://quinlangroup.slack.com/archives/CCEKJMDNF/p1553722993008200

import os
import math
import strformat
import argparse
import strutils
import hts/bam
import genoiser
import streams

proc discordant(aln:Record): bool =
  if aln.tid != aln.mate_tid:
    return true
  if aln.isize.abs > 10_000: return true

proc bad(aln:Record): bool {.inline.} =
  if aln.mapping_quality <= 5: return true
  #if aln.discordant: return true
  var f = aln.flag
  return (f.unmapped or f.qcfail or f.dup)

proc read_coverage*(aln:Record, posns:var seq[mrange]) =
  ## depthfun is an example of a `fun` that can be sent to `genoiser`.
  ## it sets reports the depth at each position
  if aln.bad: return
  #refposns(aln, posns)
  posns.add((aln.start, aln.stop, 1))

proc fragment_coverage*(aln:Record, posns:var seq[mrange]) =
  ## if true proper pair, then increment the entire fragment. otherwise, increment
  ## the start and end of each read separately.
  #if aln.mapping_quality < 5: return
  if aln.discordant or aln.flag.secondary or aln.flag.supplementary:
    read_coverage(aln, posns)
    return

  if aln.isize < 0: return
  if aln.isize == 0 and aln.flag.read1: return
  if aln.bad: return
  posns.add((aln.start, aln.start + aln.isize, 1))
  #posns.add((aln.stop, aln.mate_pos.int, 1))

proc mean[T: float32|int16](vals: var seq[T]): float32 =
  var s:float64
  for v in vals:
    s += v.float64
  s = s / vals.len.float64
  return s.float32

proc main() =

  var p = newParser("nofrag"):
    option("-c", "--chrom", help="which chromosome to normalize", default="1")
    arg("bam", nargs=1)
    arg("fasta", nargs=1)

  var opts = p.parse()
  if opts.bam == "" or opts.fasta == "":
    quit p.help & "\nfasta and bam arguments are required"

  var ibam:Bam
  if not open(ibam, opts.bam, threads=2, index=true, fai=opts.fasta):
    quit "couldn't open bam"
  var bamopts = SamField.SAM_FLAG.int or SamField.SAM_RNAME.int or SamField.SAM_POS.int or SamField.SAM_MAPQ.int or SamField.SAM_CIGAR.int or SamField.SAM_TLEN.int
  bamopts = bamopts or SamField.SAM_QNAME.int or SamField.SAM_RNEXT.int or SamField.SAM_PNEXT.int
  #opts = opts or SamField.SAM_AUX.int
  discard ibam.set_option(FormatOption.CRAM_OPT_REQUIRED_FIELDS, bamopts)
  discard ibam.set_option(FormatOption.CRAM_OPT_DECODE_MD, 0)

  var targets = ibam.hdr.targets
  var i = -1

  for k, t in targets:
    if t.name == opts.chrom:
      i = k
  if i == -1:
    quit "couldn't find chromosome:" & opts.chrom & " in bam"

  var L = targets[i].length.int
  var fragFun = Fun[int16](values:newSeq[int16](L + 1), f:fragment_coverage)
  var readFun = Fun[int16](values:newSeq[int16](L + 1), f:read_coverage)

  var fns = @[fragFun, readFun]

  if not genoiser(ibam, fns, targets[i].name, 0, L):
    quit "bad"

  stderr.write_line &"writing 'normed.{opts.chrom}.bin' and 'meaned.{opts.chrom}.bin'"
  var normed = newSeq[float32](fragFun.values.len)
  var rm = mean(readFun.values)
  var sn = newFileStream(&"normed.{opts.chrom}.bin", fmWrite)
  var sm = newFileStream(&"meaned.{opts.chrom}.bin", fmWrite)
  for i in 0..<fragFun.values.high:
    var rnorm = readFun.values[i].float32 / rm
    sm.write(rnorm)

    if fragFun.values[i].float32 == 0:
      normed[i] = 0
    else:
      normed[i] = readFun.values[i].float32 / max(1, fragFun.values[i].float32)
  sm.close

  var m = mean(normed)
  for i, v in normed:
    sn.write(v/m)

  sn.close



when isMainModule:
  main()
