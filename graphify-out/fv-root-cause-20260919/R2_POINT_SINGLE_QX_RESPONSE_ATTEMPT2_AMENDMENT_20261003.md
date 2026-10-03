# Single-qx response fixture API correction

Attempt1 terminated with AttributeError before response calculation or any endpoint solve: the producer assumed a `_observation_values` method that FVPointResearchProblem does not expose. Preserve its failed parent/log/preflight and source snapshot. It is execution failed / response not_reached, not a scientific refusal or response result.

Use the actual point-objective layout, parameters[:-1].reshape_as(problem.observation_dbz), while recalculating background via problem.contract(parameters). The fixed whitener, statistical problem, prior, control chart, solver tolerances, h values, resource limits and all numerical criteria remain unchanged. Annotate the fixture helper with the real problem type; replace the fictitious mock API and add a real fixed-problem contract regression for signed observations and theta-dependent background. Eight focused tests pass; no optimizer/response is run by those tests.

Attempt2 uses the original600s/1GiB resource guard and same registered four signed endpoints. This corrects a programming interface defect; it does not retry a numerical rejection or expand a reported scientific range.
