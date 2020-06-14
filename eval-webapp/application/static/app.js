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
                templateUrl: function (routeParams) {
                    console.log("/summary", routeParams);
                    return "summary.html";
                },
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
            .otherwise({
                redirectTo: '/summary',
            });
    }
]).directive('ngPrism', ['$interpolate', function ($interpolate) {
    return {
        restrict: 'E',
        template: `<pre><code ng-transclude></code></pre>`,
        replace: true,
        transclude: true
    };
}]).directive('ngCheck', function () {
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
        for (let idx = 0; idx < $scope.metrics.length; ++idx) {
            let metric = $scope.metrics[idx];
            let rowValues = [];
            let rowFormattedValues = [];
            let rowClasses = [];
            for (let value of $scope.values[idx]) {
                let convertedValue = metric.converter ? metric.converter(value) : value;
                let formattedValue = metric.formatter ? metric.formatter(value) : value;
                rowValues.push(convertedValue);
                rowFormattedValues.push(formattedValue);
            }
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
                if (metric.align) classes.push(metric.align, "aligned");
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
        },
        template: `
            <table class="ui celled definition table">
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
        `
    };
});

App.factory('State', ['$http', '$timeout', function ($http, $timeout) {
    let state = {
        ready: false,
        summary: null,
        examples: [],
    };
    $http.get("/static/data/eval-small.json").then(function (response) {
        state.examples = response.data.examples;
        state.summary = response.data.summary;
        state.ready = true;
    });

    return {
        getExample: function (idx) {
            return state.examples[idx];
        },
        getSummary: function () {
            return state.summary;
        },
        getCount: function () {
            return state.examples.length;
        },
        isReady: function () {
            return state.ready;
        },
        lastIndex: 1,
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
    $scope.summaryNames = State.getSummary().summary_table[0].slice(1);
    $scope.summaryValues = [];
    for (let row of State.getSummary().summary_table.slice(1))
        $scope.summaryValues.push(row.slice(1));

    function parseFrac(str) {
        let [numer, denom] = str.split(" / ");
        return numer / denom;
    }

    $scope.summaryMetrics = [
        // name, higher-is-better?, formatter, converter
        {name: "BLEU4", higherIsBetter: true, converter: parseFloat, align: "center"},
        {name: "BLEU8", higherIsBetter: true, converter: parseFloat, align: "center"},
        {name: "BLEU4 (ignoring identifiers)", higherIsBetter: true, converter: parseFloat, align: "center"},
        {name: "Unparsable function signature", higherIsBetter: false, converter: parseFrac, align: "center"},
        {name: "Correct func names", higherIsBetter: true, converter: parseFrac, align: "center"},
        {name: "Correct return types (ignoring CV)", higherIsBetter: true, converter: parseFrac, align: "center"},
        {name: "Correct return types (strict)", higherIsBetter: true, converter: parseFrac, align: "center"},
        {name: "Correct argument names", higherIsBetter: true, converter: parseFrac, align: "center"},
        {name: "Correct argument types (ignoring CV)", higherIsBetter: true, converter: parseFrac, align: "center"},
        {name: "Correct argument types (strict)", higherIsBetter: true, converter: parseFrac, align: "center"},
        {name: "Missing arguments", higherIsBetter: false, converter: parseFrac, align: "center"},
        {name: "Redundant arguments", higherIsBetter: false, converter: parseFloat, align: "center"},
        {name: "Missing string literals", higherIsBetter: false, converter: parseFloat, align: "center"},
        {name: "Redundant string literals", higherIsBetter: false, converter: parseFloat, align: "center"},
        {name: "Pointer conversion", higherIsBetter: null, align: "center"},
    ];
}]);

App.controller('ExampleCtrl', ['State', '$location', '$route', '$routeParams', '$scope', '$timeout', function (State, $location, $route, $routeParams, $scope, $timeout) {
    $scope.metricKeys = ["bleu4", "bleu8", "bleu4_no_var", "overlap_score"];

    function fixed(digits) {
        return function (x) {
            return x.toFixed(digits);
        };
    }

    $scope.metricDescriptions = [
        {name: "BLEU4", higherIsBetter: true, formatter: fixed(2), align: "right"},
        {name: "BLEU8", higherIsBetter: true, formatter: fixed(2), align: "right"},
        {name: "BLEU4 (ignoring identifiers)", higherIsBetter: true, formatter: fixed(2), align: "right"},
        {name: "Similarity score", higherIsBetter: null, formatter: fixed(3), align: "right"},
    ];

    $scope.exampleCount = State.getCount();

    function validateIndex(idx) {
        idx = parseInt(idx);
        if (!Number.isNaN(idx) && (1 <= idx && idx <= $scope.exampleCount))
            return idx;
        return null;
    }

    $scope.gotoExampleIdx = null;
    $scope.switchExample = function (idx) {
        if (idx === $scope.idx) return;
        $scope.idx = idx;
        State.lastIndex = idx;
        $location.path("/example/" + idx, true);
        $scope.gotoExampleIdx = idx;

        // Compute everything that's needed to render this example.
        $scope.example = State.getExample(idx - 1);
        $scope.srcTgt = [
            ["src", "Decompiled (source)", $scope.example.src],
            ["tgt", "Original (target)", $scope.example.tgt],
        ];

        // Compute metric comparisons.
        $scope.metricValues = [];
        $scope.predNames = [];
        for (let key of $scope.metricKeys) {
            let values = [];
            for (let data of $scope.example.preds)
                values.push(data[key]);
            $scope.metricValues.push(values);
        }
        for (let data of $scope.example.preds)
            $scope.predNames.push(data.source);

        // $timeout(function () {
        // Add syntax highlighting after digest cycle, so the code is fully-loaded.
        $scope.$$postDigest(function () {
            Prism.highlightAll();

            let src = $("#src-code").html();
            let varMap = $scope.example.var_map;
            for (const [varId, varNames] of Object.entries(varMap)) {
                let decompSpan = "<span class='decompiled-var'>" + varNames[0] + "</span>";
                let oracleSpan = "<span class='oracle-var'>" + varNames[1] + "</span>";
                src = src.replace(new RegExp(varId, 'g'), decompSpan + oracleSpan);
            }
            $("#src-code").html(src);
            $scope.updateVarName();
            console.log("Switched to example " + idx);
        });
        // }, 100);
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
}]);