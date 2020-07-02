'use strict';

let App = angular.module('App', [
    'ngRoute',
    'ngSanitize',
]).config(['$routeProvider', '$locationProvider', '$sanitizeProvider',
    function ($routeProvider, $locationProvider, $sanitizeProvider) {
        $locationProvider.hashPrefix('!');
        $sanitizeProvider.addValidAttrs(["style"]);
        $routeProvider
            .when('/summary', {
                controller: "SummaryCtrl",
                templateUrl: "summary.html",
                activeTab: "summary",
            })
            .when('/example/:id?', {
                controller: "ExampleCtrl",
                templateUrl: function (routeParams) {
                    console.log("/example", routeParams);
                    return "example.html";
                },
                activeTab: "example",
            })
            .when('/compare', {
                controller: "CompareCtrl",
                templateUrl: "compare.html",
            })
            .otherwise({
                redirectTo: '/summary',
            });
    }
]);

App.directive('ngPrism', function () {
    return {
        restrict: 'E',
        template: `<pre><code class="language-{{language}}"></code></pre>`,
        scope: {
            language: "@",
            code: "=",
            varMap: "=?",
        },
        link: {
            post: function ($scope, $element) {
                const codeElement = $element.find("code").get(0);
                $scope.$watch("code", function () {
                    // We have to manually set code contents, because Prism removes stuff that AngularJS uses to keep
                    // track of bindings.
                    let $parent = $element.parent();
                    while ($parent.attr("name") === undefined) $parent = $parent.parent();
                    const systemName = $parent.attr("name");
                    codeElement.innerHTML = $scope.code;
                    Prism.highlightElement(codeElement);

                    if ($scope.varMap) {
                        let src = codeElement.innerHTML;
                        for (const [varId, varNames] of Object.entries($scope.varMap)) {
                            let decompSpan = "<span class='decompiled-var'>" + varNames[0] + "</span>";
                            let oracleSpan = "<span class='oracle-var'>" + varNames[1] + "</span>";
                            src = src.replace(new RegExp(varId, 'g'), decompSpan + oracleSpan);
                        }
                        codeElement.innerHTML = src;
                    }
                });
            },
        },
    };
});

App.directive('ngCheck', function () {
    return {
        restrict: 'E',
        scope: {
            pred: "=",
            label: "=",
            cond: "=",
        },
        transclude: true,
        template: `
            <span ng-style="{color: cond || pred == label ? 'green' : 'red'}">
                <code>{{pred}}</code> {{cond || pred == label ? '✓' : '✗'}} <span ng-transclude></span>
            </span>
        `,
    };
});

App.controller('MetricTableCtrl', ['$scope', function ($scope) {
    function updateTable() {
        $scope.formattedValues = [];
        $scope.cellClasses = [];
        for (const metric of $scope.metrics) {
            let rowValues = $scope.values.map(dict => metric.toValue(dict[metric.key]));
            let rowFormattedValues = $scope.values.map(dict => metric.format(dict[metric.key]));
            let rowClasses = [];
            let bestValue = null, worstValue = null;
            if (metric.higherIsBetter === true) {
                bestValue = Math.max(...rowValues);
                worstValue = Math.min(...rowValues);
            } else if (metric.higherIsBetter === false) {
                bestValue = Math.min(...rowValues);
                worstValue = Math.max(...rowValues);
            }
            for (let value of rowValues) {
                let classes = [];
                if (value === bestValue) classes.push("positive");
                else if (value === worstValue) classes.push("negative");
                if ($scope.align) classes.push($scope.align, "aligned");
                rowClasses.push(classes.join(" "));
            }
            $scope.formattedValues.push(rowFormattedValues);
            $scope.cellClasses.push(rowClasses);
        }
    }

    $scope.$watch("values", updateTable);
}]).directive('ngMetricTable', function () {
    return {
        controller: 'MetricTableCtrl',
        restrict: 'E',
        scope: {
            values: "=",
            metrics: "=",
            names: "=",
            align: "@?",
        },
        template: `
            <table class="ui celled definition table metric-table">
                <thead><tr>
                    <th></th>
                    <th class="center aligned" ng-repeat="x in names">{{x}}</th>
                </tr></thead>
                <tbody>
                    <tr ng-repeat="metric in metrics" ng-init="rowIdx = $index">
                        <td>
                            <i ng-if="metric.higherIsBetter === true" class="green icon sort up"></i>
                            <i ng-if="metric.higherIsBetter === false" class="red icon sort down"></i>
                            {{metric.name}}
                        </td>
                        <td ng-repeat="value in formattedValues[rowIdx] track by $index" ng-init="colIdx = $index"
                            class="{{cellClasses[rowIdx][colIdx]}}">
                            {{value}}
                        </td>
                    </tr>
                </tbody>
            </table>
        `,
    };
});

App.directive('ngHypothesis', function () {
    return {
        restrict: 'E',
        scope: {
            data: "=",
            target: "=",
            metrics: "=",
            varMap: "=?",
        },
        templateUrl: "hypothesis.html",
    };
});

App.controller('AccordionCtrl', ['State', '$scope', '$element', function (State, $scope, $element) {
    let _active = State.getPersistenceState("accordion-" + $element.id);
    $scope.active = {
        setDefault: function (key, value) {
            if (_active[key] === undefined)
                _active[key] = value;
        },
        get: function (key) {
            return _active[key];
        },
        toggle: function (key) {
            _active[key] = !_active[key];
        },
    };
}]).directive('ngAccordion', function () {
    return {
        controller: 'AccordionCtrl',
        restrict: 'E',
        scope: {
            exclusive: "=?",
            defaultActive: "=?",
        },
        transclude: true,
        template: `
            <div class="ui styled fluid accordion" ng-transclude>
            </div>
        `,
        link: {
            pre: function ($scope, $element) {
                if (angular.isUndefined($scope.defaultActive))
                    $scope.defaultActive = false;
                if (angular.isUndefined($scope.exclusive))
                    $scope.exclusive = true;
                $scope.$$postDigest(function () {
                    $element.accordion({
                        exclusive: $scope.exclusive,
                    });
                });
            }
        },
    };
}).directive('ngAccordionEntry', function () {
    return {
        require: '^ngAccordion',
        restrict: 'E',
        scope: {
            id: "@",
            name: "@",
            defaultActive: "=?",
        },
        transclude: true,
        template: `
            <div class="title" ng-click="active.toggle(id)">
                <i class="dropdown icon"></i>
                {{name}}
            </div>
            <div class="content" ng-transclude></div>
        `,
        link: {
            pre: function ($scope, $element) {
                const $parentScope = $element.parent().scope();
                if (angular.isUndefined($scope.defaultActive))
                    $scope.defaultActive = $parentScope.defaultActive;
                if (angular.isUndefined($scope.exclusive))
                    $scope.exclusive = true;

                $scope.active = $parentScope.active;
                $scope.active.setDefault($scope.id, $scope.defaultActive);
                if ($scope.active.get($scope.id))
                    $element.find("div").addClass("active");
            },
        },
    };
});

App.factory('State', ['$http', '$timeout', function ($http, $timeout) {
    let state = {
        ready: false,
        metrics: null,
        systems: null,
        examples: [],
    };
    let _persistence = {};

    class Metric {
        constructor(metricJson) {
            this.key = metricJson.key;
            this.name = metricJson.name;
            this.higherIsBetter = metricJson.higher_is_better;
            this.formatter = metricJson.formatter || {};
            this.displayInSummary = metricJson.display_in_summary;
            this.displayInExample = metricJson.display_in_example;
        }

        toValue(value) { return value; }

        difference(lhs, rhs) { return lhs - rhs; }
    }

    class IntMetric extends Metric {
        format(value) { return value.toString(); }
    }

    class FloatMetric extends Metric {
        format(value) {
            if (this.formatter.fixed !== undefined) value = value.toFixed(this.formatter.fixed);
            return value.toString();
        }
    }

    class PortionMetric extends Metric {
        format(value) {
            return value.correct + " / " + value.total;
        }

        toValue(value) { return value.correct / value.total; }

        difference(lhs, rhs) {
            return {
                correct: lhs.correct - rhs.correct,
                total: lhs.total - rhs.total,
            };
        }
    }

    class ConfusionMatMetric extends Metric {
        format(value) {
            const tp = value.true_positive, fp = value.false_positive, fn = value.false_negative;
            return "P: " + (tp + " / " + (tp + fp)) + ", R: " + (tp + " / " + (tp + fn));
        }
    }

    const metricClass = {
        int: IntMetric,
        float: FloatMetric,
        portion: PortionMetric,
        confusion_mat: ConfusionMatMetric,
    };

    $http.get("/static/data/eval-test.json").then(function (response) {
        state.examples = response.data.examples;
        state.metrics = [];
        for (const metric of response.data.metrics)
            state.metrics.push(new metricClass[metric.type](metric));
        state.systems = response.data.systems;
        state.ready = true;
    });

    return {
        getExample: function (idx) {
            return state.examples[idx];
        },
        getMetrics: function () {
            return state.metrics;
        },
        getSystems: function () {
            return state.systems;
        },
        getCount: function () {
            return state.examples.length;
        },
        isReady: function () {
            return state.ready;
        },
        lastIndex: 1,
        getPersistenceState: function (id) {
            let obj = _persistence[id];
            if (obj === undefined)
                obj = _persistence[id] = {};
            return obj;
        }
    };
}]);

App.run(['$route', '$rootScope', '$location', function ($route, $rootScope, $location) {
    // Add option in `$location.path` to prevent reload when updating path.
    const original = $location.path;
    $location.path = function (path, preventReload) {
        if (preventReload === true) {
            const lastRoute = $route.current;
            const unsubscribe = $rootScope.$on('$locationChangeSuccess', function () {
                $route.current = lastRoute;
                unsubscribe();
            });
        }
        return original.apply($location, [path]);
    };
}]);

App.controller('MainCtrl', ['State', '$route', '$scope', '$timeout', function (State, $route, $scope, $timeout) {
    $scope.$route = $route;
    $scope.links = [
        {id: "summary", name: "Summary", url: "/summary"},
        {id: "example", name: "Browse Examples", url: "/example"},
        {id: "compare", name: "Compare Systems", url: "/compare"},
    ];

    $scope.isReady = State.isReady;
    $scope.getCount = State.getCount;

    $scope.$watch(State.isReady, function (newVal, oldVal) {
        if (newVal !== oldVal && newVal) {
            $("#loading-message").transition({
                animation: "fade",
                onComplete: function () {
                    $("#load-success-message").transition({
                        animation: "fade",
                        onComplete: function () {
                            $timeout(function () {
                                $(".success.message").transition("fade");
                            }, 1500);
                        },
                    });
                },
            });
        }
    });
}]);

App.controller('SummaryCtrl', ['State', '$scope', function (State, $scope) {
    $scope.summaryNames = State.getSystems().map(system => system.name);
    $scope.summaryValues = State.getSystems().map(system => system.metrics);
    $scope.summaryMetrics = State.getMetrics().filter(metric => metric.displayInSummary);
}]);

App.controller('ExampleCtrl', ['State', '$location', '$route', '$routeParams', '$scope', '$timeout', function (State, $location, $route, $routeParams, $scope, $timeout) {
    $scope.exampleCount = State.getCount();
    let _firstTime = true;

    function validateIndex(idx) {
        idx = parseInt(idx);
        if (!Number.isNaN(idx) && (1 <= idx && idx <= $scope.exampleCount))
            return idx;
        return null;
    }

    $scope.metrics = State.getMetrics().filter(metric => metric.displayInExample);
    $scope.systems = State.getSystems();
    $scope.systemNames = State.getSystems().map(system => system.name);

    $scope.gotoExampleIdx = null;
    $scope.switchExample = function (idx) {
        if (idx === $scope.idx) return;
        $scope.idx = idx;
        State.lastIndex = idx;
        $location.path("/example/" + idx, true);
        $scope.gotoExampleIdx = idx;

        // Compute everything that's needed to render this example.
        $scope.example = State.getExample(idx - 1);
        $scope.metricValues = $scope.systems.map(system => $scope.example.predictions[system.key].metrics);
        console.log("Switched to example " + idx);
    };
    $scope.gotoInputError = false;
    $scope.updateGoto = (function () {
        let timer = null;
        return function () {
            if (timer !== null) {
                $timeout.cancel(timer);
                timer = null;
            }
            // Instant validation.
            let idx = validateIndex($scope.gotoExampleIdx);
            if (idx !== null) {
                $scope.gotoInputError = false;
                // Delayed switch.
                timer = $timeout(function () {
                    $scope.switchExample(idx);
                }, 500);
            } else {
                $scope.gotoInputError = true;
            }
        };
    })();

    $scope.showOracleVar = false;
    $scope.updateVarName = function () {
        if ($scope.showOracleVar) {
            $(".decompiled-var").addClass("hide");
            $(".oracle-var").removeClass("hide");
        } else {
            $(".decompiled-var").removeClass("hide");
            $(".oracle-var").addClass("hide");
        }
    };

    let idx = validateIndex($routeParams.id) || State.lastIndex;
    $scope.switchExample(idx);
    // Update variable names after each digest cycle, so the code is fully-loaded.
    $scope.$watch(function () {
        $scope.$$postDigest($scope.updateVarName);
    });
}]);

App.controller('CompareCtrl', ['State', '$scope', '$timeout', function (State, $scope, $timeout) {
    $scope.$$postDigest(function () {
        $(".ui.dropdown").dropdown({
            values: [],
        });
    });
}]);