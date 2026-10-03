  @echo off
  docker run --rm -it -v "%cd%:/work" -w /work -v "C:\cs683\traces:/traces:ro" gcc:7 %*