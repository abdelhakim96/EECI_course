function results=run_public_tests(varargin)
resList{1}=modgen.containers.test.run_tests();
resList{2}=modgen.configuration.test.run_tests();
resList{3}=modgen.xml.test.run_tests;
resList{4}=modgen.common.test.run_tests();
resList{5}=modgen.struct.test.run_tests;
resList{6}=modgen.string.test.run_tests;
resList{7}=modgen.logging.test.run_tests;
resList{8}=modgen.logging.log4j.test.run_tests(varargin{:});
resList{9}=modgen.cell.test.run_tests;
resList{10}=modgen.profiling.test.run_tests;
resList{11}=modgen.pcalc.test.run_tests(varargin{:});
resList{12}=modgen.microsoft.test.run_tests;
resList{13}=modgen.graphics.test.run_tests(varargin{:});
resList{14}=modgen.io.test.run_tests(varargin{:});
resList{15}=modgen.algo.test.run_tests();
resList{16}=modgen.system.test.run_tests();
resList{17}=modgen.selfmnt.test.mlunit.run_tests();
resList{18}=modgen.dev.test.mlunit.run_tests();
results=[resList{:}];