import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { 
  TrendingUp, 
  Users, 
  Database, 
  Activity 
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export function DashboardOverview() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="h-[100px]">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div className="text-sm    font-medium>Models Trained</h2>
              <div className="flex h-8 w-8 items-center justify-center bg-blue-500/10 text-blue-500 rounded-md">
                <TrendingUp className="h-4 w-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">24</p>
            <p className="text-xs text-muted-foreground">This month</p>
          </CardContent>
          <CardFooter className="flex justify-between">
            <Badge variant="secondary">+12%</Badge>
            <Button variant="link" size="sm" className="p-1">
              View all
            </Button>
          </CardFooter>
        </Card>
        
        <Card className="h-[100px]">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-medium">Active Agents</h2>
              <div className="flex h-8 w-8 items-center justify-center bg-green-500/10 text-green-500 rounded-md">
                <Users className="h-4 w-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">12</p>
            <p className="text-xs text-muted-foreground">Running</p>
          </CardContent>
          <CardFooter className="flex justify-between">
            <Badge variant="secondary">Online</Badge>
            <Button variant="link" size="sm" className="p-1">
              Manage
            </Button>
          </CardFooter>
        </Card>
        
        <Card className="h-[100px]">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-medium">Datasets</h2>
              <div className="flex h-8 w-8 items-center justify-center bg-purple-500/10 text-purple-500 rounded-md">
                <Database className="h-4 w-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">156</p>
            <p className="text-xs text-muted-foreground">Total</p>
          </CardContent>
          <CardFooter className="flex justify-between">
            <Badge variant="secondary">+5 this week</Badge>
            <Button variant="link" size="sm" className="p-1">
              Browse
            </Button>
          </CardFooter>
        </Card>
        
        <Card className="h-[100px]">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-medium">API Requests</h2>
              <div className="flex h-8 w-8 items-center justify-center bg-indigo-500/10 text-indigo-500 rounded-md">
                <Activity className="h-4 w-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">12.4K</p>
            <p className="text-xs text-muted-foreground">Today</p>
          </CardContent>
          <CardFooter className="flex justify-between">
            <Badge variant="secondary">Peak: 3.2K/h</Badge>
            <Button variant="link" size="sm" className="p-1">
              Details
            </Button>
          </CardFooter>
        </Card>
      </div>
      
      <div className="grid gap-6 md:grid-cols-2">
        <Card className="h-[300px]">
          <CardHeader>
            <h2 className="text-lg font-semibold">System Performance</h2>
          </CardHeader>
          <CardContent>
            <div className="h-[200px]">
              {/* Placeholder for charts */}
              <div className="h-full w-full bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-lg flex items-center justify-center text-muted-foreground">
                <div className="text-center space-y-3">
                  <div className="flex items-center justify-center space-x-3">
                    <div className="w-3 h-3 bg-blue-400/50 rounded-full animate-pulse" />
                    <div className="w-3 h-3 bg-blue-400/50 rounded-full animate-pulse" />
                    <div className="w-3 h-3 bg-blue-400/50 rounded-full animate-pulse" />
                  </div>
                  <p className="mt-2 text-sm">Charts and metrics coming soon</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="h-[300px]">
          <CardHeader>
            <h2 className="text-lg font-semibold">Recent Activity</h2>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-start space-x-3">
              <div className="flex h-8 w-8 items-center justify-center bg-blue-500/10 text-blue-500 rounded-md shrink-0">
                <div className="h-4 w-4 bg-blue-500 rounded-full" />
              </div>
              <div className="flex-1 space-y-1">
                <p className="font-medium">Model training completed</p>
                <p className="text-sm text-muted-foreground">ResNet-50 finished training at 92.3% accuracy</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="flex h-8 w-8 items-center justify-center bg-green-500/10 text-green-500 rounded-md shrink-0">
                <div className="h-4 w-4 bg-green-500 rounded-full" />
              </div>
              <div className="flex-1 space-y-1">
                <p className="font-medium">New dataset uploaded</p>
                <p className="text-sm text-muted-foreground">customer_data_2024.csv (2.4 MB)</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="flex h-8 w-8 items-center justify-center bg-purple-500/10 text-purple-500 rounded-md shrink-0">
                <div className="h-4 w-4 bg-purple-500 rounded-full" />
              </div>
              <div className="flex-1 space-y-1">
                <p className="font-medium">Deployment successful</p>
                <p className="text-sm text-muted-foreground">API v2.1.0 deployed to production</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3 justify-between">
              <Button variant="outline" size="sm" className="px-3">
                View all activity
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}