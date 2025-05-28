using SafeTestsets, Neuroblox
# using Test, TestSetExtensions, SafeTestsets, Logging

Logging.global_logger(NullLogger())

@testset ExtendedTestSet "NeurobloxGUI tests" begin
  @includetests ARGS
end
